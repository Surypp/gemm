# Utility to query the compute capability of the first available GPU at
# configure time and populate GEMM_DETECTED_ARCH.
# Called optionally from the root CMakeLists if the user wants auto-detection.

function(gemm_detect_arch OUT_VAR)
    if(NOT CMAKE_CUDA_COMPILER)
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    set(_detect_src [=[
#include <cstdio>
int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("%d%d\n", prop.major, prop.minor);
}
]=])
    file(WRITE "${CMAKE_BINARY_DIR}/detect_arch.cu" "${_detect_src}")
    try_run(
        _run_result _compile_result
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/detect_arch.cu"
        CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=native"
        RUN_OUTPUT_VARIABLE _arch_output
    )
    if(_compile_result AND _run_result EQUAL 0)
        string(STRIP "${_arch_output}" _arch_output)
        set(${OUT_VAR} "${_arch_output}" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()
