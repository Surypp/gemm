#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>

// ─── BenchmarkRow ─────────────────────────────────────────────────────────────
struct BenchmarkRow {
    std::string phase;
    std::string dtype;
    int M, N, K;
    int BM, BN, BK;
    double mean_ms          = 0.0;
    double stddev_ms        = 0.0;
    double min_ms           = 0.0;
    double tflops           = 0.0;
    double pct_rocblas_peak = 0.0;
    std::string error;       // non-empty if launch failed
};

// ─── ResultTable ──────────────────────────────────────────────────────────────
struct ResultTable {
    std::vector<BenchmarkRow> rows;

    void add(BenchmarkRow r) { rows.push_back(std::move(r)); }

    // ── ASCII table to stdout ──────────────────────────────────────────────────
    void print_table(FILE* out = stdout) const {
        fprintf(out, "%-12s %-5s %5s %5s %5s %4s %4s %4s %8s %8s %8s %6s\n",
                "phase", "dtype", "M", "N", "K",
                "BM", "BN", "BK",
                "mean_ms", "TFLOPS", "min_ms", "% peak");
        fprintf(out, "%s\n", std::string(90, '-').c_str());
        for (auto& r : rows) {
            if (!r.error.empty()) continue;
            fprintf(out, "%-12s %-5s %5d %5d %5d %4d %4d %4d %8.3f %8.2f %8.3f %6.1f\n",
                    r.phase.c_str(), r.dtype.c_str(),
                    r.M, r.N, r.K,
                    r.BM, r.BN, r.BK,
                    r.mean_ms, r.tflops, r.min_ms, r.pct_rocblas_peak);
        }
    }

    // ── CSV ───────────────────────────────────────────────────────────────────
    void write_csv(const std::string& path) const {
        std::ofstream f(path);
        f << "phase,dtype,M,N,K,BM,BN,BK,mean_ms,stddev_ms,min_ms,tflops,pct_rocblas_peak,error\n";
        for (auto& r : rows) {
            if (!r.error.empty()) continue;
            f << r.phase   << ","
              << r.dtype   << ","
              << r.M       << "," << r.N << "," << r.K << ","
              << r.BM      << "," << r.BN << "," << r.BK << ","
              << std::fixed << std::setprecision(4)
              << r.mean_ms << "," << r.stddev_ms << "," << r.min_ms << ","
              << r.tflops  << ","
              << std::setprecision(2) << r.pct_rocblas_peak << ","
              << r.error   << "\n";
        }
    }

    // ── JSON ──────────────────────────────────────────────────────────────────
    void write_json(const std::string& path) const {
        std::ofstream f(path);
        f << "[\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            auto& r = rows[i];
            f << "  {\n"
              << "    \"phase\": \""      << r.phase  << "\",\n"
              << "    \"dtype\": \""      << r.dtype  << "\",\n"
              << "    \"M\": "            << r.M      << ",\n"
              << "    \"N\": "            << r.N      << ",\n"
              << "    \"K\": "            << r.K      << ",\n"
              << "    \"BM\": "           << r.BM     << ",\n"
              << "    \"BN\": "           << r.BN     << ",\n"
              << "    \"BK\": "           << r.BK     << ",\n"
              << "    \"mean_ms\": "      << std::fixed << std::setprecision(4) << r.mean_ms  << ",\n"
              << "    \"stddev_ms\": "    << r.stddev_ms << ",\n"
              << "    \"min_ms\": "       << r.min_ms << ",\n"
              << "    \"tflops\": "       << std::setprecision(4) << r.tflops << ",\n"
              << "    \"pct_peak\": "     << std::setprecision(2) << r.pct_rocblas_peak << ",\n"
              << "    \"error\": \""      << r.error  << "\"\n"
              << "  }" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        f << "]\n";
    }
};
