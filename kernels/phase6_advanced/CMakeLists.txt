gemm_add_kernel_lib(gemm_phase6a
    gemm_ldmatrix_fp16.cu
)
# ldmatrix.sync requires real SM compilation for sm_80, sm_90a, sm_120.
# Override GEMM_ARCH_LIST: this library always compiles for all three targets
# regardless of the global setting.
set_target_properties(gemm_phase6a PROPERTIES
    CUDA_ARCHITECTURES "80;90a;120"
)
target_include_directories(gemm_phase6a PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)

gemm_add_kernel_lib(gemm_phase6b
    gemm_cpasync_fp16.cu
)
# cp.async requires sm_80+; compile for all three targets.
set_target_properties(gemm_phase6b PROPERTIES
    CUDA_ARCHITECTURES "80;90a;120"
)
target_include_directories(gemm_phase6b PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
