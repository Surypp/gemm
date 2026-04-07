gemm_add_kernel_lib(gemm_phase5
    gemm_mma_ptx_fp16.cu
)
# PTX mma.sync requires real SM compilation (not compute_ virtual arch).
# -arch=sm_80 is set via GEMM_ARCH_LIST → CUDA_ARCHITECTURES in the macro.
