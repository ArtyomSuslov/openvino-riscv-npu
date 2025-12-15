set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV64_RVV1p0 ON)
set(RISCV64_SPACEMIT ON) 

set(RISCV_TOOLCHAIN_ROOT $ENV{RISCV_TOOLCHAIN_ROOT} CACHE PATH "Path to GCC for RISC-V cross compiler build directory")
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")

set(MY_ARCH_FLAGS "-march=rv64gcv_zfh_zvfh -mabi=lp64d")

set(CMAKE_C_FLAGS_INIT "${MY_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${MY_ARCH_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${MY_ARCH_FLAGS}")

set(CMAKE_C_COMPILER_TARGET "riscv64-unknown-linux-gnu")
set(CMAKE_CXX_COMPILER_TARGET "riscv64-unknown-linux-gnu")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-pthread")