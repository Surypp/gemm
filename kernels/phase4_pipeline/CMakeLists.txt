gemm_add_kernel_lib(gemm_phase4
    gemm_pipeline_fp16.cu
)
# cp.async requires sm_80+ (CUDA 11.1+)
