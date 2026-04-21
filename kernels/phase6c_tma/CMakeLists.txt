gemm_add_kernel_lib(gemm_phase6c
    gemm_tma_mmasync_fp16.cu
)
# TMA requires sm_90a+; A100 (sm_80) has no TMA hardware.
set_target_properties(gemm_phase6c PROPERTIES
    CUDA_ARCHITECTURES "90a;120"
)
target_include_directories(gemm_phase6c PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
# cuTensorMapEncodeTiled is in the driver API (libcuda.so / cuda.lib).
target_link_libraries(gemm_phase6c PUBLIC CUDA::cuda_driver)
