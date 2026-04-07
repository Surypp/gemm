#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>

// --- SwizzlePattern ---
// XOR-based index permutation that eliminates shared memory bank conflicts
// without padding.
//
// Background
// ----------
// Shared memory is banked into 32 banks of 4 bytes each (on Ampere/Hopper).
// For FP16 (2 bytes), two consecutive elements share one 4-byte bank slot.
//
// A row-major tile of shape [BM][BK] stored in SRAM:
//   element (i, j) lives at bank  (i * BK/2 + j/2) % 32   (for FP16)
//
// When threads in a warp load column j across BM consecutive rows (i=0..BM-1),
// all accesses land on the same bank → 2-way to 32-way conflict.
//
// The XOR swizzle replaces column j with:
//   j_swz = j ^ (f(i) & xor_mask)
// where f(i) changes with the row such that, for any set of 32 threads
// accessing consecutive rows, every thread hits a different bank.
//
// Parameters
// ----------
//   BK         — tile width in elements (must be power-of-2)
//   DtypeBytes — sizeof(scalar_t): 4 for FP32, 2 for FP16/BF16
//
// The XOR mask is derived automatically from BK and DtypeBytes.

template <int BK, int DtypeBytes>
struct SwizzlePattern {
    static_assert((BK & (BK - 1)) == 0, "BK must be a power of 2");
    static_assert(DtypeBytes == 2 || DtypeBytes == 4,
                  "DtypeBytes must be 2 (FP16/BF16) or 4 (FP32/TF32)");

    static constexpr int kBanks       = 32;
    static constexpr int kBankBytes   = 4;
    static constexpr int kElemsPerWord = kBankBytes / DtypeBytes;  // 2 for FP16, 1 for FP32

    static constexpr int kElemsPerLine = 128 / DtypeBytes;

    // Number of rows after which the XOR pattern repeats.
    // For FP16, BK=32:  kElemsPerWord=2, kBanks=32 → group_size = 32*2/32 = 2
    // For FP16, BK=64:  group_size = 64*2/32 = 4 ... etc.
    static constexpr int kGroupSize  = (BK * kElemsPerWord) / kBanks;
    static constexpr int kGroupShift = []() constexpr {
        int g = kGroupSize, s = 0;
        while (g > 1) { g >>= 1; ++s; }
        return s;
    }();

    // XOR mask covers the bits of the column index that are permuted.
    // For BK=32 FP16:  31 >> 1 = 15  (bits 0..3 of the half-element index)
    static constexpr int kXorMask = (BK / kElemsPerWord) - 1;

    // --- core permutation ---
    // Usage in kernel:  smem[row][permute_col(row, col)]
    __host__ __device__ __forceinline__
    static int permute_col(int row, int col) {
        int word_col = col / kElemsPerWord;
        int xor_val  = (row >> kGroupShift) & kXorMask;
        int swz_word = word_col ^ xor_val;
        return swz_word * kElemsPerWord + (col % kElemsPerWord);
    }

    // --- conflict-free check ---
    // For a warp of 32 threads accessing row `base_row + t` (t=0..31),
    // each loading column `col`, verify all resulting bank indices are distinct.
    static bool verify_conflict_free(int col = 0, int base_row = 0) {
        bool seen[32] = {};
        for (int t = 0; t < 32; ++t) {
            int row      = base_row + t;
            int swz_col  = permute_col(row, col);
            int byte_off = row * BK * DtypeBytes + swz_col * DtypeBytes;
            int bank     = (byte_off / kBankBytes) % kBanks;
            if (seen[bank]) return false;
            seen[bank] = true;
        }
        return true;
    }

    static bool verify_all(bool verbose = false) {
        for (int col = 0; col < BK; ++col) {
            for (int base_row = 0; base_row < 64; base_row += 32) {
                if (!verify_conflict_free(col, base_row)) {
                    if (verbose)
                        printf("[SwizzlePattern<%d,%d>] CONFLICT at col=%d base_row=%d\n",
                               BK, DtypeBytes, col, base_row);
                    return false;
                }
            }
        }
        return true;
    }
};

// --- aliases ---
// Swizzle = SwizzlePattern<BK, dtype_size_v<Tag>>;
// Then: int smem_col = Swizzle::permute_col(row, col);
