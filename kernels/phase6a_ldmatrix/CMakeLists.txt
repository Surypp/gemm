gemm_add_kernel_lib(gemm_phase6a
    gemm_ldmatrix_fp16.cu
)
# ldmatrix.sync requires sm_80+; compile for all three targets.
set_target_properties(gemm_phase6a PROPERTIES
    CUDA_ARCHITECTURES "80;90a;120"
)
target_include_directories(gemm_phase6a PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
