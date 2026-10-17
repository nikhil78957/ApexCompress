#pragma once

#include "common.hpp"
#include "state_table.hpp"
#include <vector>
#include <cstdint>

namespace apex {

// Hash-based context model entry (3 bytes: 16-bit tag + 8-bit state)
#pragma pack(push, 1)
struct ContextEntry {
    uint16_t tag;
    uint8_t  state;
};
#pragma pack(pop)

// Fast power-of-two hash context table with 16-bit tags
class HashContextModel {
public:
    HashContextModel(size_t log_size) : mask((1 << log_size) - 1), table(1 << log_size, {0, 0}) {}

    inline uint32_t predict(uint64_t hash) const {
        uint32_t idx = (uint32_t)(hash & mask);
        uint16_t tag = (uint16_t)(hash >> 16);
        if (table[idx].tag == tag) {
            return StateTable::instance().state_prob[table[idx].state];
        }
        return PROB_HALF; // Neutral 50/50 when unseen
    }

    inline void update(uint64_t hash, int bit) {
        uint32_t idx = (uint32_t)(hash & mask);
        uint16_t tag = (uint16_t)(hash >> 16);
        if (table[idx].tag == tag) {
            table[idx].state = StateTable::instance().next_state[table[idx].state][bit];
        } else {
            table[idx].tag = tag;
            table[idx].state = StateTable::instance().next_state[0][bit];
        }
    }

    void reset() {
        std::fill(table.begin(), table.end(), ContextEntry{0, 0});
    }

private:
    uint32_t mask;
    std::vector<ContextEntry> table;
};

// Multi-Scale Context Modeling Engine
class MultiContextModel {
public:
    MultiContextModel() 
        : order0(256, 0),
          order1(65536, 0),
          order2(18), // 256K entries
          order3(19), // 512K entries
          order4(19), // 512K entries
          order6(19)  // 512K entries
    {}

    // Queries order-0, order-1, order-2, order-3, order-4, order-6 models
    inline void predict_all(uint8_t cxt, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4, uint8_t b5, uint8_t b6,
                            uint32_t* probs_out) const {
        // Order-0: within-byte bit position
        probs_out[0] = StateTable::instance().state_prob[order0[cxt]];

        // Order-1: previous byte + bit prefix
        uint32_t o1_idx = ((uint32_t)b1 << 8) | cxt;
        probs_out[1] = StateTable::instance().state_prob[order1[o1_idx]];

        // Order-2: previous 2 bytes + cxt
        uint64_t h2 = (((uint64_t)b1 << 16) | ((uint64_t)b2 << 8) | cxt) * 0x517cc1b727220a95ull;
        probs_out[2] = order2.predict(h2);

        // Order-3: previous 3 bytes + cxt
        uint64_t h3 = (h2 ^ ((uint64_t)b3 << 24)) * 0x9e3779b97f4a7c15ull;
        probs_out[3] = order3.predict(h3);

        // Order-4: previous 4 bytes + cxt
        uint64_t h4 = (h3 ^ ((uint64_t)b4 << 32)) * 0xbf58476d1ce4e5b9ull;
        probs_out[4] = order4.predict(h4);

        // Order-6: previous 6 bytes + cxt
        uint64_t h6 = (h4 ^ ((uint64_t)b5 << 40) ^ ((uint64_t)b6 << 48)) * 0x94d049bb133111ebull;
        probs_out[5] = order6.predict(h6);
    }

    // Updates all models with the observed bit
    inline void update_all(uint8_t cxt, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4, uint8_t b5, uint8_t b6, int bit) {
        order0[cxt] = StateTable::instance().next_state[order0[cxt]][bit];

        uint32_t o1_idx = ((uint32_t)b1 << 8) | cxt;
        order1[o1_idx] = StateTable::instance().next_state[order1[o1_idx]][bit];

        uint64_t h2 = (((uint64_t)b1 << 16) | ((uint64_t)b2 << 8) | cxt) * 0x517cc1b727220a95ull;
        order2.update(h2, bit);

        uint64_t h3 = (h2 ^ ((uint64_t)b3 << 24)) * 0x9e3779b97f4a7c15ull;
        order3.update(h3, bit);

        uint64_t h4 = (h3 ^ ((uint64_t)b4 << 32)) * 0xbf58476d1ce4e5b9ull;
        order4.update(h4, bit);

        uint64_t h6 = (h4 ^ ((uint64_t)b5 << 40) ^ ((uint64_t)b6 << 48)) * 0x94d049bb133111ebull;
        order6.update(h6, bit);
    }

    void reset() {
        std::fill(order0.begin(), order0.end(), 0);
        std::fill(order1.begin(), order1.end(), 0);
        order2.reset();
        order3.reset();
        order4.reset();
        order6.reset();
    }

private:
    std::vector<uint8_t> order0;
    std::vector<uint8_t> order1;
    HashContextModel order2;
    HashContextModel order3;
    HashContextModel order4;
    HashContextModel order6;
};

} // namespace apex
