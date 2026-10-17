#pragma once

#include "common.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace apex {

// Neural Context Mixer (Adaptive Logistic Perceptron)
// Blends multiple context models using stochastic gradient descent in the logit domain
class ContextMixer {
public:
    static constexpr int NUM_MODELS = 8;
    static constexpr int NUM_WEIGHT_SETS = 256; // Conditioned on bit prefix (cxt)
    static constexpr int WEIGHT_SCALE_BITS = 8;  // Fixed-point scale for weights (256 = 1.0)
    static constexpr int LEARNING_RATE_SHIFT = 11; // Controls gradient step size

    ContextMixer() : weights(NUM_WEIGHT_SETS * (NUM_MODELS + 1), 0) {
        // Initialize weights with balanced prior
        for (int c = 0; c < NUM_WEIGHT_SETS; ++c) {
            int* w = &weights[c * (NUM_MODELS + 1)];
            w[0] = 0;   // Bias
            w[1] = 128; // Order-0
            w[2] = 256; // Order-1
            w[3] = 384; // Order-2
            w[4] = 384; // Order-3
            w[5] = 256; // Order-4
            w[6] = 256; // Order-6
            w[7] = 512; // Match model (high prior weight)
            w[8] = 256; // Text model
        }
    }

    // Predicts the mixed probability using logistic dot product
    inline uint32_t mix(const uint32_t* probs, uint8_t cxt, int16_t* logits_out) const {
        const int* w = &weights[cxt * (NUM_MODELS + 1)];
        int y = w[0]; // Start with bias

        for (int i = 0; i < NUM_MODELS; ++i) {
            logits_out[i] = to_logit(probs[i]);
            y += (w[i + 1] * (int)logits_out[i]) >> WEIGHT_SCALE_BITS;
        }

        return squash(y);
    }

    // Updates weights via online gradient descent (LMS rule)
    inline void train(uint8_t cxt, const int16_t* logits, uint32_t p_mixed, int bit) {
        int* w = &weights[cxt * (NUM_MODELS + 1)];
        
        // Error = (target - p_mixed) in [-4096, 4096]
        int target = bit ? PROB_SCALE : 0;
        int error = target - (int)p_mixed;

        // Bias update
        int bias_grad = (error << 2);
        w[0] = std::clamp(w[0] + (bias_grad >> (LEARNING_RATE_SHIFT - 2)), -16384, 16384);

        // Feature weight updates
        for (int i = 0; i < NUM_MODELS; ++i) {
            int grad = (error * (int)logits[i]);
            int delta = grad >> LEARNING_RATE_SHIFT;
            w[i + 1] = std::clamp(w[i + 1] + delta, -32767, 32767);
        }
    }

    void reset() {
        for (int c = 0; c < NUM_WEIGHT_SETS; ++c) {
            int* w = &weights[c * (NUM_MODELS + 1)];
            w[0] = 0;
            w[1] = 128;
            w[2] = 256;
            w[3] = 384;
            w[4] = 384;
            w[5] = 256;
            w[6] = 256;
            w[7] = 512;
            w[8] = 256;
        }
    }

private:
    std::vector<int> weights;
};

} // namespace apex
