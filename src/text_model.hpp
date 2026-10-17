#pragma once

#include "common.hpp"
#include "state_table.hpp"
#include "context_model.hpp"
#include <cstdint>
#include <cctype>

namespace apex {

// Specialized Text and Word-Level Context Model
// Models natural language words, code tokens, whitespace, and capitalization patterns
class TextModel {
public:
    TextModel() 
        : word_table(18), // 256K entries
          prev_word_hash(0),
          curr_word_hash(0),
          in_word(false)
    {}

    // Queries text model prediction for the current bit
    inline uint32_t predict(uint8_t cxt) const {
        uint64_t h = ((uint64_t)prev_word_hash << 32) | ((uint64_t)curr_word_hash << 8) | cxt;
        h *= 0x517cc1b727220a95ull;
        return word_table.predict(h);
    }

    // Updates text model state after a bit is observed
    inline void update(uint8_t cxt, int bit) {
        uint64_t h = ((uint64_t)prev_word_hash << 32) | ((uint64_t)curr_word_hash << 8) | cxt;
        h *= 0x517cc1b727220a95ull;
        word_table.update(h, bit);
    }

    // Called when a full byte is observed to advance word boundary tracking
    inline void add_byte(uint8_t b) {
        if (std::isalnum(b) || b == '_') {
            in_word = true;
            curr_word_hash = (curr_word_hash * 33) ^ std::tolower(b);
        } else {
            if (in_word) {
                prev_word_hash = curr_word_hash;
                curr_word_hash = b;
                in_word = false;
            } else {
                curr_word_hash = (curr_word_hash * 33) ^ b;
            }
        }
    }

    void reset() {
        word_table.reset();
        prev_word_hash = 0;
        curr_word_hash = 0;
        in_word = false;
    }

private:
    HashContextModel word_table;
    uint32_t prev_word_hash;
    uint32_t curr_word_hash;
    bool in_word;
};

} // namespace apex
