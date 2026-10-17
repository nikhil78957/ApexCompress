#pragma once

#include "common.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace apex {

// Compact 256-state adaptive probability estimator (1 byte per context entry)
// Represents counts of 0s and 1s with non-stationary exponential decay
class StateTable {
public:
    uint8_t next_state[256][2];
    uint16_t state_prob[256];

    static const StateTable& instance() {
        static StateTable st;
        return st;
    }

private:
    StateTable() {
        // Enumerate (n0, n1) pairs to fit in 256 states
        struct Pair { int n0, n1; };
        std::vector<Pair> pairs;
        pairs.push_back({0, 0}); // State 0: neutral 50/50

        // Generate states for small exact counts
        for (int total = 1; total <= 24; ++total) {
            for (int n1 = 0; n1 <= total; ++n1) {
                int n0 = total - n1;
                pairs.push_back({n0, n1});
                if (pairs.size() >= 256) break;
            }
            if (pairs.size() >= 256) break;
        }

        // Fill remaining states up to 256 with higher confidence points
        int step = 2;
        while (pairs.size() < 256) {
            pairs.push_back({step * 15, 0});
            if (pairs.size() < 256) pairs.push_back({0, step * 15});
            step++;
            if (step > 20) break;
        }
        while (pairs.size() < 256) {
            pairs.push_back({10, 10});
        }

        // Compute probabilities and state transitions
        for (int s = 0; s < 256; ++s) {
            int n0 = pairs[s].n0;
            int n1 = pairs[s].n1;

            // Krichevsky-Trofimov estimator
            double p = ((double)n1 + 0.5) / ((double)n0 + (double)n1 + 1.0);
            int int_p = (int)std::round(p * (double)PROB_SCALE);
            state_prob[s] = (uint16_t)std::clamp(int_p, 1, (int)PROB_SCALE - 1);

            // Compute next states with aging/decay when count gets high
            for (int bit = 0; bit <= 1; ++bit) {
                int next_n0 = n0 + (bit == 0 ? 1 : 0);
                int next_n1 = n1 + (bit == 1 ? 1 : 0);

                if (next_n0 + next_n1 > 28) {
                    // Decay old counts by ~15% to maintain adaptivity to non-stationary data
                    next_n0 = (next_n0 * 7) / 8;
                    next_n1 = (next_n1 * 7) / 8;
                }

                // Find closest existing state
                double target_p = ((double)next_n1 + 0.5) / ((double)next_n0 + (double)next_n1 + 1.0);
                int best_state = 0;
                double best_dist = 1e9;

                for (size_t cand = 0; cand < pairs.size(); ++cand) {
                    double cand_p = ((double)pairs[cand].n1 + 0.5) / ((double)pairs[cand].n0 + (double)pairs[cand].n1 + 1.0);
                    int count_diff = std::abs((pairs[cand].n0 + pairs[cand].n1) - (next_n0 + next_n1));
                    double dist = std::abs(cand_p - target_p) * 100.0 + count_diff;
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_state = (int)cand;
                    }
                }
                next_state[s][bit] = (uint8_t)best_state;
            }
        }
    }
};

} // namespace apex
