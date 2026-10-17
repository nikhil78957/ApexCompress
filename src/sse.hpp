#pragma once

#include "common.hpp"
#include <vector>
#include <algorithm>

namespace apex {

// Secondary Symbol Estimator (SSE)
// Refines and calibrates the neural mixer's probability using empirical bit frequencies
class SecondarySymbolEstimator {
public:
    static constexpr int BINS = 32;
    static constexpr int NUM_CONTEXTS = 512; // (cxt & 0x1F) << 4 | (b1 >> 4)

    SecondarySymbolEstimator() : table(NUM_CONTEXTS * (BINS + 1)) {
        // Initialize table so initially p_out == p_in (identity calibration)
        for (int c = 0; c < NUM_CONTEXTS; ++c) {
            for (int b = 0; b <= BINS; ++b) {
                int init_prob = std::clamp((b * PROB_SCALE) / BINS, 1u, PROB_SCALE - 1u);
                table[c * (BINS + 1) + b] = (uint16_t)init_prob;
            }
        }
    }

    // Refines input probability p_in into calibrated output probability
    inline uint32_t calibrate(uint32_t p_in, uint8_t cxt, uint8_t b1) const {
        int ctx = ((cxt & 0x1F) << 4) | (b1 >> 4);
        int bin = (int)(p_in * BINS / PROB_SCALE);
        bin = std::clamp(bin, 0, BINS - 1);

        int frac = (int)(p_in * BINS % PROB_SCALE); // Interpolation weight
        int base_idx = ctx * (BINS + 1) + bin;

        uint32_t p0 = table[base_idx];
        uint32_t p1 = table[base_idx + 1];

        // Linear interpolation between quantization bins
        uint32_t interpolated = (p0 * (PROB_SCALE - frac) + p1 * frac) >> PROB_BITS;
        return std::clamp(interpolated, 1u, PROB_SCALE - 1u);
    }

    // Updates the calibration table with the actual observed bit
    inline void update(uint32_t p_in, uint8_t cxt, uint8_t b1, int bit) {
        int ctx = ((cxt & 0x1F) << 4) | (b1 >> 4);
        int bin = (int)(p_in * BINS / PROB_SCALE);
        bin = std::clamp(bin, 0, BINS - 1);

        int target = bit ? (PROB_SCALE - 1) : 1;
        int base_idx = ctx * (BINS + 1) + bin;

        // Adaptive update with stationary rate (1/32)
        int err0 = target - (int)table[base_idx];
        table[base_idx] = (uint16_t)std::clamp((int)table[base_idx] + (err0 >> 5), 1, (int)PROB_SCALE - 1);

        int err1 = target - (int)table[base_idx + 1];
        table[base_idx + 1] = (uint16_t)std::clamp((int)table[base_idx + 1] + (err1 >> 5), 1, (int)PROB_SCALE - 1);
    }

    void reset() {
        for (int c = 0; c < NUM_CONTEXTS; ++c) {
            for (int b = 0; b <= BINS; ++b) {
                int init_prob = std::clamp((b * PROB_SCALE) / BINS, 1u, PROB_SCALE - 1u);
                table[c * (BINS + 1) + b] = (uint16_t)init_prob;
            }
        }
    }

private:
    std::vector<uint16_t> table;
};

} // namespace apex
