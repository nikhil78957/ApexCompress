#pragma once

#include "common.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace apex {

// LZ-style Match Predictor: detects repeated sequences of bytes in history
// and predicts following bits with confidence proportional to match length
class MatchModel {
public:
    static constexpr size_t HISTORY_SIZE = 1 << 18; // 256 KB history window
    static constexpr size_t HISTORY_MASK = HISTORY_SIZE - 1;
    static constexpr size_t HASH_SIZE = 1 << 18;    // 256 K hash slots
    static constexpr size_t HASH_MASK = HASH_SIZE - 1;

    MatchModel() 
        : history(HISTORY_SIZE, 0),
          pos_table(HASH_SIZE, 0),
          head(0),
          match_pos(0),
          match_len(0),
          expected_byte(0)
    {}

    // Query prediction for the current bit
    // bit_pos is 0 for MSB (bit 7), up to 7 for LSB (bit 0)
    inline uint32_t predict(int bit_pos) const {
        if (match_len < 4) {
            return PROB_HALF; // Insufficient match history, neutral 50/50
        }

        int expected_bit = (expected_byte >> (7 - bit_pos)) & 1;
        // Confidence scales smoothly with match length up to ~99.5%
        int conf = std::min(32, match_len) * 60;
        
        if (expected_bit == 1) {
            return PROB_HALF + conf;
        } else {
            return PROB_HALF - conf;
        }
    }

    // Called when a full byte is completed to update history and detect new matches
    inline void add_byte(uint8_t byte) {
        history[head & HISTORY_MASK] = byte;
        head++;

        // Update match finder using 4-byte rolling hash
        if (head >= 4) {
            uint8_t b0 = history[(head - 1) & HISTORY_MASK];
            uint8_t b1 = history[(head - 2) & HISTORY_MASK];
            uint8_t b2 = history[(head - 3) & HISTORY_MASK];
            uint8_t b3 = history[(head - 4) & HISTORY_MASK];

            uint32_t h = ((uint32_t)b3 << 24) | ((uint32_t)b2 << 16) | ((uint32_t)b1 << 8) | b0;
            h = (h ^ (h >> 16)) * 0x45d9f3b;
            uint32_t slot = h & HASH_MASK;

            size_t prev = pos_table[slot];
            pos_table[slot] = head;

            if (match_len > 0) {
                // Check if current active match is still continuing
                if (history[(match_pos) & HISTORY_MASK] == byte) {
                    match_pos++;
                    match_len++;
                    expected_byte = history[match_pos & HISTORY_MASK];
                    return;
                } else {
                    match_len = 0; // Match broke
                }
            }

            // Test if previous position is a valid match of at least 4 bytes
            if (prev > 0 && (head - prev) < HISTORY_SIZE && (head - prev) > 4) {
                size_t p1 = head - 1;
                size_t p2 = prev - 1;
                int len = 0;
                while (len < 255 && p1 > 0 && p2 > 0 && 
                       history[p1 & HISTORY_MASK] == history[p2 & HISTORY_MASK]) {
                    len++;
                    p1--;
                    p2--;
                }
                if (len >= 4) {
                    match_pos = prev;
                    match_len = len;
                    expected_byte = history[match_pos & HISTORY_MASK];
                }
            }
        }
    }

    void reset() {
        std::fill(history.begin(), history.end(), 0);
        std::fill(pos_table.begin(), pos_table.end(), 0);
        head = 0;
        match_pos = 0;
        match_len = 0;
        expected_byte = 0;
    }

private:
    std::vector<uint8_t> history;
    std::vector<uint32_t> pos_table;
    size_t head;
    size_t match_pos;
    int match_len;
    uint8_t expected_byte;
};

} // namespace apex
