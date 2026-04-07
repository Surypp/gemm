#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// --- dtype tags ---
// Hardware tensor cores always accumulate in FP32 regardless of input precision,
// so accum_t = float for all reduced-precision tags.

struct FP32Tag {
    using scalar_t = float;
    using accum_t  = float;
    static constexpr const char* name = "fp32";
};

struct FP16Tag {
    using scalar_t = __half;
    using accum_t  = float;  // wmma / mma.sync accumulate in FP32
    static constexpr const char* name = "fp16";
};

struct BF16Tag {
    using scalar_t = __nv_bfloat16;
    using accum_t  = float;
    static constexpr const char* name = "bf16";
};

template <typename Tag>
constexpr int dtype_size_v = static_cast<int>(sizeof(typename Tag::scalar_t));

// --- layout ---
enum class Layout { RowMajor, ColMajor };

// --- GemmDesc ---
// All pointers are device pointers.
// lda, ldb, ldc are leading dimensions in elements (not bytes).
// For RowMajor A (M×K): lda >= K.  For ColMajor A (M×K): lda >= M.
template <typename Tag, Layout LayoutA = Layout::RowMajor,
                        Layout LayoutB = Layout::RowMajor>
struct GemmDesc {
    using scalar_t = typename Tag::scalar_t;
    using accum_t  = typename Tag::accum_t;

    int M, N, K;

    scalar_t  alpha;       // C = alpha*(A*B) + beta*C
    accum_t   beta;

    const scalar_t* A;     // M×K
    const scalar_t* B;     // K×N
    accum_t*        C;     // M×N  (in/out)

    int lda, ldb, ldc;
};

template <typename Tag>
using GemmDescRowMajor = GemmDesc<Tag, Layout::RowMajor, Layout::RowMajor>;
