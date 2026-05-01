gemm_add_kernel_lib(gemm_phase6e
    gemm_fp8_e4m3.cu
)
# FP8 mma.sync requires sm_89+ (Ada/Hopper/Blackwell).
# sm_80 (A100) has no FP8 tensor core support.
set_target_properties(gemm_phase6e PROPERTIES
    CUDA_ARCHITECTURES "90a;120"
)
target_include_directories(gemm_phase6e PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
# cuTensorMapEncodeTiled is in the driver API.
target_link_libraries(gemm_phase6e PUBLIC CUDA::cuda_driver)
