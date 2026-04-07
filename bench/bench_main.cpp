#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

#include "gemm/cuda_helpers.cuh"
#include "bench_results.hpp"

// Forward declarations
ResultTable run_suite_fp16(int warmup, int iters);

namespace bench_cublas {
double measure_cublas_fp16_tflops(int M, int N, int K, int iters);
double measure_cublas_fp32_tflops(int M, int N, int K, int iters);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --dtype   fp16|fp32        (default: fp16)\n"
        "  --warmup  N                (default: 5)\n"
        "  --iters   N                (default: 20)\n"
        "  --csv     PATH             write CSV results\n"
        "  --json    PATH             write JSON results\n"
        "  --device  N                GPU device id (default: 0)\n"
        "  --help                     show this message\n",
        prog);
}

int main(int argc, char** argv) {
    // --- args ---
    std::string dtype   = "fp16";
    int warmup          = 5;
    int iters           = 20;
    int device_id       = 0;
    std::string csv_path, json_path;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--help"))   { print_usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--dtype"))  { dtype     = argv[++i]; }
        else if (!strcmp(argv[i], "--warmup")) { warmup    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--iters"))  { iters     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--device")) { device_id = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--csv"))    { csv_path  = argv[++i]; }
        else if (!strcmp(argv[i], "--json"))   { json_path = argv[++i]; }
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }

    // --- device info ---
    CUDA_CHECK(cudaSetDevice(device_id));
    auto info = gemm::query_device(device_id);
    gemm::print_device_info(info);
    printf("\n");

    // --- run suite ---
    ResultTable table;
    if (dtype == "fp16") {
        table = run_suite_fp16(warmup, iters);
    } else {
        fprintf(stderr, "dtype '%s' not yet supported in the suite\n", dtype.c_str());
        return 1;
    }

    // --- output ---
    printf("\n=== Results ===\n");
    table.print_table();

    if (!csv_path.empty())  table.write_csv(csv_path);
    if (!json_path.empty()) table.write_json(json_path);

    if (!csv_path.empty())  printf("CSV  → %s\n", csv_path.c_str());
    if (!json_path.empty()) printf("JSON → %s\n", json_path.c_str());

    return 0;
}
