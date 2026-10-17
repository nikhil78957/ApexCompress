#pragma once

#include "common.hpp"
#include <vector>
#include <cstdint>

namespace apex {

// Dmitry Subbotin 32-bit Adaptive Binary Range Coder
// Carry-free, branch-predictable, guaranteed exact lossless representation
class RangeEncoder {
public:
    std::vector<uint8_t> buffer;

    RangeEncoder() : low(0), range(0xFFFFFFFF) {
        buffer.reserve(65536);
    }

    inline void encode(int bit, uint32_t p1) {
        uint32_t bound = (range >> PROB_BITS) * (PROB_SCALE - p1);
        if (bit == 0) {
            range = bound;
        } else {
            low = (low + bound) & 0xFFFFFFFF;
            range -= bound;
        }

        while ((low ^ ((low + range) & 0xFFFFFFFF)) < 0x01000000u) {
            buffer.push_back((uint8_t)(low >> 24));
            low = (low << 8) & 0xFFFFFFFF;
            range = (range << 8) & 0xFFFFFFFF;
        }

        while (range < 0x00010000u) {
            buffer.push_back((uint8_t)(low >> 24));
            range = ((-low) & 0xFFFF) << 8;
            low = (low << 8) & 0xFFFFFFFF;
        }
    }

    void flush() {
        for (int i = 0; i < 4; ++i) {
            buffer.push_back((uint8_t)(low >> 24));
            low = (low << 8) & 0xFFFFFFFF;
        }
    }

private:
    uint32_t low;
    uint32_t range;
};

class RangeDecoder {
public:
    RangeDecoder(const uint8_t* data, size_t size) 
        : src(data), src_size(size), src_pos(0), low(0), range(0xFFFFFFFF), code(0) 
    {
        for (int i = 0; i < 4; ++i) {
            code = ((code << 8) | read_byte()) & 0xFFFFFFFF;
        }
    }

    inline int decode(uint32_t p1) {
        uint32_t bound = (range >> PROB_BITS) * (PROB_SCALE - p1);
        int bit;

        if (code < ((low + bound) & 0xFFFFFFFF)) {
            bit = 0;
            range = bound;
        } else {
            bit = 1;
            low = (low + bound) & 0xFFFFFFFF;
            range -= bound;
        }

        while ((low ^ ((low + range) & 0xFFFFFFFF)) < 0x01000000u) {
            code = ((code << 8) | read_byte()) & 0xFFFFFFFF;
            low = (low << 8) & 0xFFFFFFFF;
            range = (range << 8) & 0xFFFFFFFF;
        }

        while (range < 0x00010000u) {
            code = ((code << 8) | read_byte()) & 0xFFFFFFFF;
            range = ((-low) & 0xFFFF) << 8;
            low = (low << 8) & 0xFFFFFFFF;
        }

        return bit;
    }

private:
    const uint8_t* src;
    size_t src_size;
    size_t src_pos;
    uint32_t low;
    uint32_t range;
    uint32_t code;

    inline uint8_t read_byte() {
        if (src_pos < src_size) {
            return src[src_pos++];
        }
        return 0;
    }
};

} // namespace apex
