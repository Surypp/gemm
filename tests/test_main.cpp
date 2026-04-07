#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "gemm/error_check.cuh"
#include "gemm/cuda_helpers.cuh"

int main(int argc, char** argv) {
    // Print GPU info before running any test
    CUDA_CHECK(cudaSetDevice(0));
    auto info = gemm::query_device(0);
    gemm::print_device_info(info);
    printf("\n");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
