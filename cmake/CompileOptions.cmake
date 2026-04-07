# Shared NVCC flags — included by root CMakeLists.txt and referenced in each
# per-phase CMakeLists via target_compile_options.
# Per-phase files may APPEND to GEMM_NVCC_FLAGS but must not clear it.

if(MSVC)
    set(_host_warn_flags -Xcompiler=/W3 -Xcompiler=/wd4100 -Xcompiler=/wd4127 -Xcompiler=/Zc:preprocessor)
else()
    set(_host_warn_flags -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-unused-parameter)
endif()

set(GEMM_NVCC_FLAGS
    --expt-relaxed-constexpr        # Allow constexpr in device code
    --expt-extended-lambda          # Allow __device__ lambdas
    -lineinfo                       # Preserve source line info for Nsight Compute
    ${_host_warn_flags}
    CACHE INTERNAL "Shared NVCC compile flags"
)

# Debug build adds device-side bounds checking
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND GEMM_NVCC_FLAGS -G -DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    list(APPEND GEMM_NVCC_FLAGS -O2)
else()
    # Release
    list(APPEND GEMM_NVCC_FLAGS -O3)
endif()

# Helper macro used in every per-phase CMakeLists.txt
# Usage: gemm_add_kernel_lib(TARGET src1.cu src2.cu)
# Creates a static library with the shared flags and the project include dirs.
macro(gemm_add_kernel_lib TARGET)
    add_library(${TARGET} STATIC ${ARGN})
    target_include_directories(${TARGET}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${TARGET}
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_NVCC_FLAGS}>
            # Print register/smem usage per kernel — useful during development
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES "${GEMM_ARCH_LIST}"
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_link_libraries(${TARGET} PUBLIC CUDA::cudart CUDA::cublas)
endmacro()
