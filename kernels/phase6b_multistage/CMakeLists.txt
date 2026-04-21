gemm_add_kernel_lib(gemm_phase6b
    gemm_cpasync_fp16.cu
)
# cp.async (LDGSTS on SM120) requires sm_80+; compile for all three targets.
set_target_properties(gemm_phase6b PROPERTIES
    CUDA_ARCHITECTURES "80;90a;120"
)
target_include_directories(gemm_phase6b PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
