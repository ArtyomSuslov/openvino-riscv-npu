// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_fullyconnected.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "csinn/csi_nn.h"
#include "csinn_data_structure.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/shl/shl_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "rvv/rvv.h"
#include "ime/ime.h"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {
MemoryPtr prepareWeightMemory(const MemoryPtr weightsMemory, const ExecutorContext::CPtr context) {
    DEBUG_LOG("ShlFCExecutor: prepack weights");

    // Проверка типа весов
    auto prec = weightsMemory->getDescPtr()->getPrecision();
    
    // Если это INT8, мы не используем float упаковку и не кэшируем
    if (prec == ov::element::i8) {
        MemoryPtr _ptr = 
            std::make_shared<Memory>(context->getEngine(),
                                     intel_cpu::CpuBlockedMemoryDesc(prec, weightsMemory->getShape()));
        cpu_parallel_memcpy(_ptr->getData(), weightsMemory->getData(), weightsMemory->getSize());
        return _ptr;
    }

    auto create = [&]() {
        const auto& weiDesc = weightsMemory->getDescPtr();
        MemoryPtr _ptr =
            std::make_shared<Memory>(context->getEngine(),
                                     intel_cpu::CpuBlockedMemoryDesc(ov::element::f32, weightsMemory->getShape()));
        cpu_parallel_memcpy(_ptr->getData(), weightsMemory->getData(), weightsMemory->getSize());
        DEBUG_LOG("ShlFCExecutor: cache miss, perform packing");
        const auto repack_wei = ShlTensor(ShlSession(),
                                          precisionToShlDataType(weiDesc->getPrecision()),
                                          getShlDataLayoutByMemoryDesc(weiDesc, true),
                                          weiDesc->getShape().getStaticDims(),
                                          _ptr->getData());
        shl_rvv_fc_gemm_reorder_weight_fp32(repack_wei.get());
        return _ptr;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const auto& wgtDims = weightsMemory->getStaticDims();
        std::string format = "gemm_shl_" + std::to_string(wgtDims[0]) + "_" + std::to_string(wgtDims[1]);
        const std::string string_hash = format + "_" + std::to_string(weightsMemory->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(weightsMemory->getData()));
        DEBUG_LOG("ShlFCExecutor: findOrCreate, string_hash: ", string_hash);
        return static_cast<MemoryPtr>(*weightCache->findOrCreate(string_hash, create));
    }

    DEBUG_LOG("ShlFCExecutor: Weights cache is not available");
    return create();
}
}  // namespace

bool ShlFCExecutor::supports(const FCConfig& config) {
    if (config.attrs.weightsNonTransposed) {
        DEBUG_LOG("ShlFCExecutor: weightsNonTransposed is not supported!");
        return false;
    }

    if (!config.attrs.postOps.empty()) {
        DEBUG_LOG("ShlFCExecutor: PostOps are not supported");
        return false;
    }

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);

    // Логика проверки типов:
    // 1. Либо всё F32
    bool is_f32_mode = all_of(ov::element::f32, srcDesc->getPrecision(), weiDesc->getPrecision(), dstDesc->getPrecision());
    
    // 2. Либо Вход I8, Веса I8, Выход F32
    // В XML MatMul имеет input I8, output F32.
    bool is_int8_mode = (srcDesc->getPrecision() == ov::element::i8) && 
                        (weiDesc->getPrecision() == ov::element::i8) && 
                        (dstDesc->getPrecision() == ov::element::f32);

    if (!is_f32_mode && !is_int8_mode) {
        DEBUG_LOG("ShlFCExecutor: unsupported precision combination");
        return false;
    }

    if (config.attrs.withBias) {
        const auto& biaDesc = config.descs.at(ARG_BIAS);
        if (biaDesc->getPrecision() != ov::element::f32) {
            DEBUG_LOG("ShlFCExecutor: supports only f32 bias");
            return false;
        }

        const auto& biasDims = biaDesc->getShape().getStaticDims();
        const auto& outDims = dstDesc->getShape().getDims();
        const bool isByChannel = biasDims.back() == outDims.back();
        if (!isByChannel || !std::all_of(biasDims.begin(), biasDims.end() - 1, [](const Dim dim) {
                return dim == 1;
            })) {
            DEBUG_LOG("ShlFCExecutor: only 'by channel' bias is supported");
            return false;
        }
    }

    return true;
}

ShlFCExecutor::ShlFCExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context)
    : packedWeights(prepareWeightMemory(memory.at(ARG_WEI), context)) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    // Определяем режим работы
    is_int8 = (srcDesc->getPrecision() == ov::element::i8);

    // Allocate Shl session
    sess = ShlSession();

    // Allocate Shl tensors
    src = ShlTensor(sess, precisionToShlDataType(srcDesc->getPrecision()), getShlDataLayoutByMemoryDesc(srcDesc));
    src.get()->dtype = CSINN_DTYPE_A_INT8_W_INT8_O_FLOAT32;

    wei = ShlTensor(sess,
                    precisionToShlDataType(weiDesc->getPrecision()),
                    getShlDataLayoutByMemoryDesc(weiDesc, true),
                    weiDesc->getShape().getStaticDims());
    
    dst = ShlTensor(sess, precisionToShlDataType(dstDesc->getPrecision()), getShlDataLayoutByMemoryDesc(dstDesc));

    if (attrs.withBias) {
        const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
        bias = ShlTensor(sess,
                         precisionToShlDataType(biasDesc->getPrecision()),
                         getShlDataLayoutByMemoryDesc(biasDesc),
                         biasDesc->getShape().getStaticDims());
        with_bias = true;
    } else if (is_int8) {
        VectorDims bias_shape = dst.getShape();
        csinn_layout_enum bias_layout = getShlDataLayoutByMemoryDesc(memory.at(ARG_DST)->getDescPtr());

        // Выделяем память и обнуляем
        size_t num_elements = 1;
        for (auto d : bias_shape) num_elements *= d;

        // SHL обычно хранит float* для float32
        void* bias_data = calloc(num_elements, sizeof(float));
        OPENVINO_ASSERT(bias_data, "Failed to allocate memory for bias");
        bias = ShlTensor(sess, CSINN_DTYPE_FLOAT32, bias_layout, bias_shape, bias_data);

    } else {
        bias = ShlTensor(sess);
    }

    wei.setData(packedWeights->getData());
    if (with_bias) {
        bias.setData(memory.at(ARG_BIAS)->getData());
    }

    // Init FC params
    params = ShlFCParams(sess, CSINN_RVV);

    if (is_int8) {
        // 1. Устанавливаем тип квантования
        auto* p = static_cast<csinn_fc_params*>(params.get());
        p->base.quant_type = CSINN_QUANT_INT8_ASYM_W_SYM_TO_F32;
        p->base.api = CSINN_IME;

        // 2. Выделяем и заполняем qinfo для ВХОДА (Input)
        // Если scale=1, zp=0, то input обрабатывается как есть.
        src.get()->qinfo = (struct csinn_quant_info*)malloc(sizeof(struct csinn_quant_info));
        src.get()->quant_channel = 1;
        src.get()->qinfo[0].scale = 1.0f;
        src.get()->qinfo[0].zero_point = 0;

        // 3. Выделяем и заполняем qinfo для ВЕСОВ (Weights)
        auto& oc = weiDesc->getShape().getStaticDims()[0]; 
        wei.get()->quant_channel = oc;
        wei.get()->qinfo = (struct csinn_quant_info*)malloc(oc * sizeof(struct csinn_quant_info));
        
        for (size_t i = 0; i < oc; i++) {
            wei.get()->qinfo[i].scale = 1.0f;
            wei.get()->qinfo[i].zero_point = 0;
        }

        void* original_ptr = wei.getData();
        
        // 4. Вызываем INIT
        // Библиотека сама должна вызвать функцию перепаковки весов (reorder), 
        // которая изменит wei->data и wei->dtype.
        int ret = csinn_fullyconnected_init(src.get(),
                                            dst.get(),
                                            wei.get(),
                                            bias.get(),
                                            p);
        
        if (ret != CSINN_TRUE) {
            OPENVINO_THROW("ShlFCExecutor: failed to init INT8 FC");
        }

        void* new_ptr = wei.getData();
        if (new_ptr != original_ptr) {
            m_reordered_wei_buffer.reset(new_ptr, free);
        }

    } else {
        // Старый F32 init
        OPENVINO_ASSERT(csinn_fullyconnected_init(src.get(),
                                                  dst.get(),
                                                  wei.get(),
                                                  bias.get(),
                                                  static_cast<csinn_fc_params*>(params.get())) == CSINN_TRUE,
                        "ShlFCExecutor: failed to init FC");
    }
}

bool ShlFCExecutor::update(const MemoryArgs& memory) {
    // Weights and Bias have static shapes - no need to update them here
    src = src.cloneWithNewShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims());
    dst = dst.cloneWithNewShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims());

    const auto src_shape = src.getShape();
    const auto dst_shape = dst.getShape();
    dim_M = std::accumulate(dst_shape.rbegin() + 1, dst_shape.rend(), static_cast<size_t>(1), std::multiplies<>());
    dim_In = src_shape.back();
    dim_Out = dst_shape.back();
    LDA = dim_In * memory.at(ARG_SRC)->getPrecision().size();
    LDC = dim_Out * memory.at(ARG_DST)->getPrecision().size();

    return true;
}

void ShlFCExecutor::execute(const MemoryArgs& memory) {
    
    if (is_int8) {
        // Для теста просто запускаем в одном потоке
        // Обновляем указатели на данные (input может меняться каждый раз, output тоже)
        src.setData(memory.at(ARG_SRC)->getData());
        dst.setData(memory.at(ARG_DST)->getData());
        
        // bias и wei уже установлены в init (wei перепакован)
        
        // ВАЖНО: update() вызывается перед execute и обновляет размеры src/dst в ShlTensor.
        // Но csinn_fullyconnected может требовать актуальных размеров в тензорах.
        
        int ret = csinn_fullyconnected(src.get(),
                                       dst.get(),
                                       wei.get(),
                                       bias.get(),
                                       static_cast<csinn_fc_params*>(params.get()));
                                       
        OPENVINO_ASSERT(ret == CSINN_TRUE, "ShlFCExecutor: failed to execute INT8");
        return;
    }

    // Старая логика для F32
    wei.setData(packedWeights->getData());
    if (with_bias) {
        bias.setData(memory.at(ARG_BIAS)->getData());
    }

    const auto nthreads = std::min(static_cast<int>(dim_M), parallel_get_max_threads());
    parallel_nt(nthreads, [&](const int ithr, const int nthr) {
        size_t dim_M0 = 0;
        size_t dim_M1 = 0;
        splitter(dim_M, nthr, ithr, dim_M0, dim_M1);

        const auto M = dim_M1 - dim_M0;
        auto src_tensor = src.cloneWithNewShape(ov::Shape{M, dim_In});
        auto dst_tensor = dst.cloneWithNewShape(ov::Shape{M, dim_Out});
        src_tensor.setData(reinterpret_cast<uint8_t*>(memory.at(ARG_SRC)->getData()) + dim_M0 * LDA);
        dst_tensor.setData(reinterpret_cast<uint8_t*>(memory.at(ARG_DST)->getData()) + dim_M0 * LDC);

        OPENVINO_ASSERT(csinn_fullyconnected(src_tensor.get(),
                                             dst_tensor.get(),
                                             wei.get(),
                                             bias.get(),
                                             static_cast<csinn_fc_params*>(params.get())) == CSINN_TRUE,
                        "ShlFCExecutor: failed to execute");
    });
}

}  // namespace ov::intel_cpu
