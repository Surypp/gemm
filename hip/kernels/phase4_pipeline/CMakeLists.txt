find_package(OpenMP REQUIRED)
find_package(rocm_smi REQUIRED)
find_package(rocwmma REQUIRED)

gemm_hip_add_kernel_lib(gemm_hip_phase4 gemm_pipeline_fp16.hip)
target_link_libraries(gemm_hip_phase4 PUBLIC roc::rocwmma OpenMP::OpenMP_CXX rocm_smi64)
