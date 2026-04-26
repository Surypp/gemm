#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "gemm/error_check.hpp"
#include "gemm/hip_helpers.hpp"

int main(int argc, char** argv) {
    HIP_CHECK(hipSetDevice(0));
    auto info = gemm::query_device(0);
    gemm::print_device_info(info);
    printf("\n");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
