set(GEMM_HIP_ARCH "gfx1200" CACHE STRING "AMD GPU target")
set(GEMM_HIP_FLAGS
    -O3 -gline-tables-only --offload-arch=${GEMM_HIP_ARCH}
    -Wall -Wextra -Wno-unused-parameter
    CACHE INTERNAL ""
)
macro(gemm_hip_add_kernel_lib TARGET)
    add_library(${TARGET} STATIC ${ARGN})
    target_include_directories(${TARGET} PUBLIC
        ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:${GEMM_HIP_FLAGS}>)
    set_target_properties(${TARGET} PROPERTIES
        HIP_ARCHITECTURES "${GEMM_HIP_ARCH}")
    target_link_libraries(${TARGET} PUBLIC hip::host roc::rocblas)
endmacro()
