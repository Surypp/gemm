find_package(OpenMP REQUIRED)
find_package(rocm_smi REQUIRED)
find_package(rocwmma REQUIRED)

gemm_hip_add_kernel_lib(gemm_hip_phase3 gemm_wmma_fp16.hip)
target_link_libraries(gemm_hip_phase3 PUBLIC roc::rocwmma OpenMP::OpenMP_CXX rocm_smi64)
