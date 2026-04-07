gemm_add_kernel_lib(gemm_phase3
    gemm_wmma_fp16.cu
)
# wmma requires sm_80+ — a build targeting only sm_75 would fail here.
# The root CMake default is "80;90" so this is safe.
