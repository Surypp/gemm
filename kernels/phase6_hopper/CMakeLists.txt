# Phase 6: Hopper-specific kernels (wgmma + TMA)
# Only compiled when GEMM_ENABLE_HOPPER=ON and sm_90 is in GEMM_ARCH_LIST.

# Override CUDA_ARCHITECTURES to use sm_90a (the 'a' suffix is required for
# wgmma and TMA instructions — sm_90 without 'a' does not expose these).
add_library(gemm_phase6 STATIC gemm_hopper.cu)
target_include_directories(gemm_phase6
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(gemm_phase6
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_NVCC_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
# sm_90a instead of sm_90 to enable Hopper-only instructions
set_target_properties(gemm_phase6 PROPERTIES
    CUDA_ARCHITECTURES "90a"
)
target_link_libraries(gemm_phase6 PUBLIC CUDA::cudart CUDA::cublas)
