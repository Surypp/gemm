# tools/CMakeLists.txt
# Diagnostic tools for SM120 / SM90a.  Enabled via -DGEMM_ENABLE_TOOLS=ON.
# Not part of the test suite; these are standalone executables for one-shot probes.

# tma_swizzle_probe: determine the actual TMA SWIZZLE_64B layout on the current GPU.
# Requires TMA hardware → sm_90a (GH200) or sm_120 (RTX 5080).
add_executable(tma_swizzle_probe tma_swizzle_probe.cu)
set_target_properties(tma_swizzle_probe PROPERTIES
    CUDA_ARCHITECTURES "90a;120"
)
target_include_directories(tma_swizzle_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels
)
# cuTensorMapEncodeTiled is in the driver API.
target_link_libraries(tma_swizzle_probe PRIVATE CUDA::cuda_driver)
