#pragma once

#include "common.hpp"
#include "range_coder.hpp"
#include "state_table.hpp"
#include "context_model.hpp"
#include "match_model.hpp"
#include "text_model.hpp"
#include "mixer.hpp"
#include "sse.hpp"

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace apex {

class ApexCompressor {
public:
    // Compress raw input buffer into Apex format
    static std::vector<uint8_t> compress(const uint8_t* input, size_t input_size) {
        // Calculate CRC32 of uncompressed data
        uint32_t original_crc = CRC32::calculate(input, input_size);

        // Initialize range encoder
        RangeEncoder enc;

        // Initialize context models, mixer, and SSE
        MultiContextModel ctx_model;
        MatchModel match_model;
        TextModel text_model;
        ContextMixer mixer;
        SecondarySymbolEstimator sse;

        uint8_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
        uint32_t probs[ContextMixer::NUM_MODELS];
        int16_t logits[ContextMixer::NUM_MODELS];

        for (size_t i = 0; i < input_size; ++i) {
            uint8_t target_byte = input[i];
            uint8_t cxt = 1;

            for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
                int bit = (target_byte >> (7 - bit_pos)) & 1;

                // 1. Query predictions from all independent context models
                ctx_model.predict_all(cxt, b1, b2, b3, b4, b5, b6, probs);
                probs[6] = match_model.predict(bit_pos);
                probs[7] = text_model.predict(cxt);

                // 2. Blend predictions with neural logistic mixer
                uint32_t p_mixed = mixer.mix(probs, cxt, logits);

                // 3. Calibrate probability with Secondary Symbol Estimator
                uint32_t p_final = sse.calibrate(p_mixed, cxt, b1);

                // 4. Encode bit into range coder bitstream
                enc.encode(bit, p_final);

                // 5. Adapt weights and model states online
                mixer.train(cxt, logits, p_mixed, bit);
                sse.update(p_mixed, cxt, b1, bit);
                ctx_model.update_all(cxt, b1, b2, b3, b4, b5, b6, bit);
                text_model.update(cxt, bit);

                cxt = (cxt << 1) | bit;
            }

            match_model.add_byte(target_byte);
            text_model.add_byte(target_byte);

            b6 = b5; b5 = b4; b4 = b3; b3 = b2; b2 = b1; b1 = target_byte;
        }

        enc.flush();

        // Construct final payload with header
        ApexHeader header;
        header.magic = APEX_MAGIC;
        header.version = APEX_VERSION;
        header.reserved[0] = 0;
        header.reserved[1] = 0;
        header.reserved[2] = 0;
        header.uncompressed_size = (uint64_t)input_size;
        header.crc32 = original_crc;

        std::vector<uint8_t> output;
        output.resize(sizeof(ApexHeader) + enc.buffer.size());
        std::memcpy(output.data(), &header, sizeof(ApexHeader));
        std::memcpy(output.data() + sizeof(ApexHeader), enc.buffer.data(), enc.buffer.size());

        return output;
    }

    // Decompress Apex format buffer back into original uncompressed data
    static std::vector<uint8_t> decompress(const uint8_t* compressed_data, size_t compressed_size) {
        if (compressed_size < sizeof(ApexHeader)) {
            throw std::runtime_error("Invalid Apex compressed stream: file too small for header.");
        }

        ApexHeader header;
        std::memcpy(&header, compressed_data, sizeof(ApexHeader));

        if (header.magic != APEX_MAGIC) {
            throw std::runtime_error("Invalid Apex magic header identifier.");
        }
        if (header.version != APEX_VERSION) {
            throw std::runtime_error("Unsupported Apex format version.");
        }

        size_t payload_offset = sizeof(ApexHeader);
        size_t payload_size = compressed_size - payload_offset;

        RangeDecoder dec(compressed_data + payload_offset, payload_size);

        MultiContextModel ctx_model;
        MatchModel match_model;
        TextModel text_model;
        ContextMixer mixer;
        SecondarySymbolEstimator sse;

        uint8_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
        uint32_t probs[ContextMixer::NUM_MODELS];
        int16_t logits[ContextMixer::NUM_MODELS];

        std::vector<uint8_t> output(header.uncompressed_size);

        for (size_t i = 0; i < header.uncompressed_size; ++i) {
            uint8_t cxt = 1;

            for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
                // 1. Query predictions identically to compressor
                ctx_model.predict_all(cxt, b1, b2, b3, b4, b5, b6, probs);
                probs[6] = match_model.predict(bit_pos);
                probs[7] = text_model.predict(cxt);

                // 2. Blend predictions with neural logistic mixer
                uint32_t p_mixed = mixer.mix(probs, cxt, logits);

                // 3. Calibrate probability with SSE
                uint32_t p_final = sse.calibrate(p_mixed, cxt, b1);

                // 4. Decode bit from range coder bitstream
                int bit = dec.decode(p_final);

                // 5. Adapt weights and model states online in identical lockstep
                mixer.train(cxt, logits, p_mixed, bit);
                sse.update(p_mixed, cxt, b1, bit);
                ctx_model.update_all(cxt, b1, b2, b3, b4, b5, b6, bit);
                text_model.update(cxt, bit);

                cxt = (cxt << 1) | bit;
            }

            uint8_t decoded_byte = (uint8_t)(cxt & 0xFF);
            output[i] = decoded_byte;

            match_model.add_byte(decoded_byte);
            text_model.add_byte(decoded_byte);

            b6 = b5; b5 = b4; b4 = b3; b3 = b2; b2 = b1; b1 = decoded_byte;
        }

        // 100% Exact Verification: check CRC32
        uint32_t decompressed_crc = CRC32::calculate(output.data(), output.size());
        if (decompressed_crc != header.crc32) {
            throw std::runtime_error("Lossless integrity check failed: CRC32 mismatch!");
        }

        return output;
    }
};

} // namespace apex
