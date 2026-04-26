find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(gemm_hip_tests
    test_main.cpp
    test_phase0.cpp
    test_phase1.cpp
    test_phase2.cpp
    test_phase3.cpp
)

target_include_directories(gemm_hip_tests PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gemm_hip_tests PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${GEMM_HIP_FLAGS}>
)

set_target_properties(gemm_hip_tests PROPERTIES
    HIP_ARCHITECTURES "${GEMM_HIP_ARCH}"
)

target_link_libraries(gemm_hip_tests PRIVATE
    gemm_hip_phase0
    gemm_hip_phase1
    gemm_hip_phase2
    gemm_hip_phase3
    hip::host
    roc::rocblas
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(gemm_hip_tests)
