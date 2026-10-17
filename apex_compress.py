#!/usr/bin/env python3
"""
ApexCompress: An Advanced Multi-Context Neural Mixing Lossless Compression Engine
Reference pure Python implementation matching the C++ binary container specification.
"""

import sys
import os
import math
import struct
import zlib
import time
from typing import Tuple, List

# --- Constants & Fixed-Point Mathematics ---
PROB_BITS = 12
PROB_SCALE = 1 << PROB_BITS  # 4096
PROB_HALF = PROB_SCALE >> 1   # 2048

LOGIT_SCALE = 64
SQUASH_TABLE_SIZE = 4096
SQUASH_OFFSET = 2048

# Precompute logit and squash lookup tables
LOGIT_TABLE = [0] * PROB_SCALE
for p in range(PROB_SCALE):
    p_clamp = max(1.0, min(float(PROB_SCALE - 1), float(p)))
    ratio = p_clamp / (float(PROB_SCALE) - p_clamp)
    val = round(math.log(ratio) * LOGIT_SCALE)
    LOGIT_TABLE[p] = max(-2047, min(2047, int(val)))

SQUASH_TABLE = [0] * SQUASH_TABLE_SIZE
for i in range(SQUASH_TABLE_SIZE):
    y = i - SQUASH_OFFSET
    prob = PROB_SCALE / (1.0 + math.exp(-y / LOGIT_SCALE))
    SQUASH_TABLE[i] = max(1, min(PROB_SCALE - 1, int(round(prob))))

def to_logit(p: int) -> int:
    p = max(1, min(PROB_SCALE - 1, p))
    return LOGIT_TABLE[p]

def squash(y: int) -> int:
    idx = y + SQUASH_OFFSET
    idx = max(0, min(SQUASH_TABLE_SIZE - 1, idx))
    return SQUASH_TABLE[idx]

# --- 256-State Adaptive Probability Table ---
class StateTable:
    def __init__(self):
        pairs = [(0, 0)]
        for total in range(1, 25):
            for n1 in range(total + 1):
                n0 = total - n1
                pairs.append((n0, n1))
                if len(pairs) >= 256:
                    break
            if len(pairs) >= 256:
                break
        step = 2
        while len(pairs) < 256:
            pairs.append((step * 15, 0))
            if len(pairs) < 256:
                pairs.append((0, step * 15))
            step += 1
            if step > 20:
                break
        while len(pairs) < 256:
            pairs.append((10, 10))

        self.next_state = [[0, 0] for _ in range(256)]
        self.state_prob = [0] * 256

        for s in range(256):
            n0, n1 = pairs[s]
            p = (n1 + 0.5) / (n0 + n1 + 1.0)
            self.state_prob[s] = max(1, min(PROB_SCALE - 1, int(round(p * PROB_SCALE))))

            for bit in (0, 1):
                next_n0 = n0 + (1 if bit == 0 else 0)
                next_n1 = n1 + (1 if bit == 1 else 0)
                if next_n0 + next_n1 > 28:
                    next_n0 = (next_n0 * 7) // 8
                    next_n1 = (next_n1 * 7) // 8

                target_p = (next_n1 + 0.5) / (next_n0 + next_n1 + 1.0)
                best_state = 0
                best_dist = 1e9
                for cand in range(len(pairs)):
                    cand_p = (pairs[cand][1] + 0.5) / (pairs[cand][0] + pairs[cand][1] + 1.0)
                    cdiff = abs((pairs[cand][0] + pairs[cand][1]) - (next_n0 + next_n1))
                    dist = abs(cand_p - target_p) * 100.0 + cdiff
                    if dist < best_dist:
                        best_dist = dist
                        best_state = cand
                self.next_state[s][bit] = best_state

STATE_TABLE = StateTable()

# --- Dmitry Subbotin 32-bit Binary Range Coder ---
class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.buffer = bytearray()

    def encode(self, bit: int, p1: int):
        bound = (self.range >> PROB_BITS) * (PROB_SCALE - p1)
        if bit == 0:
            self.range = bound
        else:
            self.low = (self.low + bound) & 0xFFFFFFFF
            self.range -= bound

        while (self.low ^ ((self.low + self.range) & 0xFFFFFFFF)) < 0x01000000:
            self.buffer.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF

        while self.range < 0x00010000:
            self.buffer.append((self.low >> 24) & 0xFF)
            self.range = ((-self.low) & 0xFFFF) << 8
            self.low = (self.low << 8) & 0xFFFFFFFF

    def flush(self):
        for _ in range(4):
            self.buffer.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF

class RangeDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.code = ((self.code << 8) | b) & 0xFFFFFFFF

    def decode(self, p1: int) -> int:
        bound = (self.range >> PROB_BITS) * (PROB_SCALE - p1)
        if self.code < ((self.low + bound) & 0xFFFFFFFF):
            bit = 0
            self.range = bound
        else:
            bit = 1
            self.low = (self.low + bound) & 0xFFFFFFFF
            self.range -= bound

        while (self.low ^ ((self.low + self.range) & 0xFFFFFFFF)) < 0x01000000:
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.code = ((self.code << 8) | b) & 0xFFFFFFFF
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF

        while self.range < 0x00010000:
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.code = ((self.code << 8) | b) & 0xFFFFFFFF
            self.range = ((-self.low) & 0xFFFF) << 8
            self.low = (self.low << 8) & 0xFFFFFFFF

        return bit

# --- Context Models ---
class HashContextModel:
    def __init__(self, log_size: int):
        self.mask = (1 << log_size) - 1
        self.tags = [0] * (1 << log_size)
        self.states = [0] * (1 << log_size)

    def predict(self, h: int) -> int:
        idx = h & self.mask
        tag = (h >> 16) & 0xFFFF
        if self.tags[idx] == tag:
            return STATE_TABLE.state_prob[self.states[idx]]
        return PROB_HALF

    def update(self, h: int, bit: int):
        idx = h & self.mask
        tag = (h >> 16) & 0xFFFF
        if self.tags[idx] == tag:
            self.states[idx] = STATE_TABLE.next_state[self.states[idx]][bit]
        else:
            self.tags[idx] = tag
            self.states[idx] = STATE_TABLE.next_state[0][bit]

class MultiContextModel:
    def __init__(self):
        self.order0 = [0] * 256
        self.order1 = [0] * 65536
        self.order2 = HashContextModel(17)
        self.order3 = HashContextModel(18)
        self.order4 = HashContextModel(18)
        self.order6 = HashContextModel(18)

    def predict_all(self, cxt: int, b1: int, b2: int, b3: int, b4: int, b5: int, b6: int, probs: List[int]):
        probs[0] = STATE_TABLE.state_prob[self.order0[cxt]]
        o1_idx = (b1 << 8) | cxt
        probs[1] = STATE_TABLE.state_prob[self.order1[o1_idx]]
        h2 = ((b1 << 16) | (b2 << 8) | cxt) * 0x517cc1b727220a95
        probs[2] = self.order2.predict(h2)
        h3 = (h2 ^ (b3 << 24)) * 0x9e3779b97f4a7c15
        probs[3] = self.order3.predict(h3)
        h4 = (h3 ^ (b4 << 32)) * 0xbf58476d1ce4e5b9
        probs[4] = self.order4.predict(h4)
        h6 = (h4 ^ (b5 << 40) ^ (b6 << 48)) * 0x94d049bb133111eb
        probs[5] = self.order6.predict(h6)

    def update_all(self, cxt: int, b1: int, b2: int, b3: int, b4: int, b5: int, b6: int, bit: int):
        self.order0[cxt] = STATE_TABLE.next_state[self.order0[cxt]][bit]
        o1_idx = (b1 << 8) | cxt
        self.order1[o1_idx] = STATE_TABLE.next_state[self.order1[o1_idx]][bit]
        h2 = ((b1 << 16) | (b2 << 8) | cxt) * 0x517cc1b727220a95
        self.order2.update(h2, bit)
        h3 = (h2 ^ (b3 << 24)) * 0x9e3779b97f4a7c15
        self.order3.update(h3, bit)
        h4 = (h3 ^ (b4 << 32)) * 0xbf58476d1ce4e5b9
        self.order4.update(h4, bit)
        h6 = (h4 ^ (b5 << 40) ^ (b6 << 48)) * 0x94d049bb133111eb
        self.order6.update(h6, bit)

class MatchModel:
    HISTORY_SIZE = 1 << 18
    HISTORY_MASK = HISTORY_SIZE - 1
    HASH_SIZE = 1 << 18
    HASH_MASK = HASH_SIZE - 1

    def __init__(self):
        self.history = bytearray(self.HISTORY_SIZE)
        self.pos_table = [0] * self.HASH_SIZE
        self.head = 0
        self.match_pos = 0
        self.match_len = 0
        self.expected_byte = 0

    def predict(self, bit_pos: int) -> int:
        if self.match_len < 4:
            return PROB_HALF
        expected_bit = (self.expected_byte >> (7 - bit_pos)) & 1
        conf = min(32, self.match_len) * 60
        return (PROB_HALF + conf) if expected_bit == 1 else (PROB_HALF - conf)

    def add_byte(self, byte: int):
        self.history[self.head & self.HISTORY_MASK] = byte
        self.head += 1

        if self.head >= 4:
            b0 = self.history[(self.head - 1) & self.HISTORY_MASK]
            b1 = self.history[(self.head - 2) & self.HISTORY_MASK]
            b2 = self.history[(self.head - 3) & self.HISTORY_MASK]
            b3 = self.history[(self.head - 4) & self.HISTORY_MASK]

            h = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
            h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
            slot = h & self.HASH_MASK

            prev = self.pos_table[slot]
            self.pos_table[slot] = self.head

            if self.match_len > 0:
                if self.history[self.match_pos & self.HISTORY_MASK] == byte:
                    self.match_pos += 1
                    self.match_len += 1
                    self.expected_byte = self.history[self.match_pos & self.HISTORY_MASK]
                    return
                else:
                    self.match_len = 0

            if prev > 0 and 4 < (self.head - prev) < self.HISTORY_SIZE:
                p1 = self.head - 1
                p2 = prev - 1
                length = 0
                while length < 255 and p1 > 0 and p2 > 0 and self.history[p1 & self.HISTORY_MASK] == self.history[p2 & self.HISTORY_MASK]:
                    length += 1
                    p1 -= 1
                    p2 -= 1
                if length >= 4:
                    self.match_pos = prev
                    self.match_len = length
                    self.expected_byte = self.history[self.match_pos & self.HISTORY_MASK]

class TextModel:
    def __init__(self):
        self.word_table = HashContextModel(17)
        self.prev_word_hash = 0
        self.curr_word_hash = 0
        self.in_word = False

    def predict(self, cxt: int) -> int:
        h = (((self.prev_word_hash << 32) | (self.curr_word_hash << 8) | cxt) * 0x517cc1b727220a95) & 0xFFFFFFFFFFFFFFFF
        return self.word_table.predict(h)

    def update(self, cxt: int, bit: int):
        h = (((self.prev_word_hash << 32) | (self.curr_word_hash << 8) | cxt) * 0x517cc1b727220a95) & 0xFFFFFFFFFFFFFFFF
        self.word_table.update(h, bit)

    def add_byte(self, b: int):
        ch = chr(b) if b < 128 else ''
        if ch.isalnum() or ch == '_':
            self.in_word = True
            self.curr_word_hash = ((self.curr_word_hash * 33) ^ ord(ch.lower())) & 0xFFFFFFFF
        else:
            if self.in_word:
                self.prev_word_hash = self.curr_word_hash
                self.curr_word_hash = b
                self.in_word = False
            else:
                self.curr_word_hash = ((self.curr_word_hash * 33) ^ b) & 0xFFFFFFFF

# --- Neural Context Mixer ---
class ContextMixer:
    NUM_MODELS = 8
    WEIGHT_SCALE_BITS = 8
    LEARNING_RATE_SHIFT = 11

    def __init__(self):
        # 256 weight vectors of size 9 (1 bias + 8 models)
        self.weights = []
        for _ in range(256):
            self.weights.append([0, 128, 256, 384, 384, 256, 256, 512, 256])

    def mix(self, probs: List[int], cxt: int, logits: List[int]) -> int:
        w = self.weights[cxt]
        y = w[0]
        for i in range(self.NUM_MODELS):
            logits[i] = to_logit(probs[i])
            y += (w[i + 1] * logits[i]) >> self.WEIGHT_SCALE_BITS
        return squash(y)

    def train(self, cxt: int, logits: List[int], p_mixed: int, bit: int):
        w = self.weights[cxt]
        target = PROB_SCALE if bit else 0
        error = target - p_mixed
        bias_grad = error << 2
        w[0] = max(-16384, min(16384, w[0] + (bias_grad >> (self.LEARNING_RATE_SHIFT - 2))))
        for i in range(self.NUM_MODELS):
            delta = (error * logits[i]) >> self.LEARNING_RATE_SHIFT
            w[i + 1] = max(-32767, min(32767, w[i + 1] + delta))

# --- Secondary Symbol Estimator (SSE) ---
class SecondarySymbolEstimator:
    BINS = 32
    NUM_CONTEXTS = 512

    def __init__(self):
        self.table = []
        for _ in range(self.NUM_CONTEXTS):
            row = []
            for b in range(self.BINS + 1):
                init_prob = max(1, min(PROB_SCALE - 1, (b * PROB_SCALE) // self.BINS))
                row.append(init_prob)
            self.table.append(row)

    def calibrate(self, p_in: int, cxt: int, b1: int) -> int:
        ctx = ((cxt & 0x1F) << 4) | (b1 >> 4)
        bin_idx = (p_in * self.BINS) // PROB_SCALE
        bin_idx = max(0, min(self.BINS - 1, bin_idx))
        frac = (p_in * self.BINS) % PROB_SCALE
        row = self.table[ctx]
        p0 = row[bin_idx]
        p1 = row[bin_idx + 1]
        interpolated = (p0 * (PROB_SCALE - frac) + p1 * frac) >> PROB_BITS
        return max(1, min(PROB_SCALE - 1, interpolated))

    def update(self, p_in: int, cxt: int, b1: int, bit: int):
        ctx = ((cxt & 0x1F) << 4) | (b1 >> 4)
        bin_idx = (p_in * self.BINS) // PROB_SCALE
        bin_idx = max(0, min(self.BINS - 1, bin_idx))
        target = PROB_SCALE - 1 if bit else 1
        row = self.table[ctx]
        row[bin_idx] = max(1, min(PROB_SCALE - 1, row[bin_idx] + ((target - row[bin_idx]) >> 5)))
        row[bin_idx + 1] = max(1, min(PROB_SCALE - 1, row[bin_idx + 1] + ((target - row[bin_idx + 1]) >> 5)))

# --- Container Header ---
APEX_MAGIC = b'APEX'
APEX_VERSION = 1
HEADER_FORMAT = '<4sBB3sQI'  # magic, version, reserved, size, crc32
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# --- Compressor & Decompressor Engine ---
def compress(data: bytes) -> bytes:
    original_crc = zlib.crc32(data) & 0xFFFFFFFF
    enc = RangeEncoder()

    ctx_model = MultiContextModel()
    match_model = MatchModel()
    text_model = TextModel()
    mixer = ContextMixer()
    sse = SecondarySymbolEstimator()

    b1 = b2 = b3 = b4 = b5 = b6 = 0
    probs = [0] * ContextMixer.NUM_MODELS
    logits = [0] * ContextMixer.NUM_MODELS

    for target_byte in data:
        cxt = 1
        for bit_pos in range(8):
            bit = (target_byte >> (7 - bit_pos)) & 1

            ctx_model.predict_all(cxt, b1, b2, b3, b4, b5, b6, probs)
            probs[6] = match_model.predict(bit_pos)
            probs[7] = text_model.predict(cxt)

            p_mixed = mixer.mix(probs, cxt, logits)
            p_final = sse.calibrate(p_mixed, cxt, b1)

            enc.encode(bit, p_final)

            mixer.train(cxt, logits, p_mixed, bit)
            sse.update(p_mixed, cxt, b1, bit)
            ctx_model.update_all(cxt, b1, b2, b3, b4, b5, b6, bit)
            text_model.update(cxt, bit)

            cxt = (cxt << 1) | bit

        match_model.add_byte(target_byte)
        text_model.add_byte(target_byte)
        b6, b5, b4, b3, b2, b1 = b5, b4, b3, b2, b1, target_byte

    enc.flush()

    header = struct.pack(HEADER_FORMAT, APEX_MAGIC, APEX_VERSION, 0, b'\x00\x00\x00', len(data), original_crc)
    return header + bytes(enc.buffer)

def decompress(compressed: bytes) -> bytes:
    if len(compressed) < HEADER_SIZE:
        raise ValueError("Invalid Apex stream: file too small for header.")

    magic, version, _, _, uncompressed_size, expected_crc = struct.unpack(HEADER_FORMAT, compressed[:HEADER_SIZE])
    if magic != APEX_MAGIC:
        raise ValueError("Invalid Apex header magic bytes.")
    if version != APEX_VERSION:
        raise ValueError("Unsupported Apex version.")

    dec = RangeDecoder(compressed[HEADER_SIZE:])

    ctx_model = MultiContextModel()
    match_model = MatchModel()
    text_model = TextModel()
    mixer = ContextMixer()
    sse = SecondarySymbolEstimator()

    b1 = b2 = b3 = b4 = b5 = b6 = 0
    probs = [0] * ContextMixer.NUM_MODELS
    logits = [0] * ContextMixer.NUM_MODELS

    output = bytearray(uncompressed_size)

    for i in range(uncompressed_size):
        cxt = 1
        for bit_pos in range(8):
            ctx_model.predict_all(cxt, b1, b2, b3, b4, b5, b6, probs)
            probs[6] = match_model.predict(bit_pos)
            probs[7] = text_model.predict(cxt)

            p_mixed = mixer.mix(probs, cxt, logits)
            p_final = sse.calibrate(p_mixed, cxt, b1)

            bit = dec.decode(p_final)

            mixer.train(cxt, logits, p_mixed, bit)
            sse.update(p_mixed, cxt, b1, bit)
            ctx_model.update_all(cxt, b1, b2, b3, b4, b5, b6, bit)
            text_model.update(cxt, bit)

            cxt = (cxt << 1) | bit

        decoded_byte = cxt & 0xFF
        output[i] = decoded_byte

        match_model.add_byte(decoded_byte)
        text_model.add_byte(decoded_byte)
        b6, b5, b4, b3, b2, b1 = b5, b4, b3, b2, b1, decoded_byte

    actual_crc = zlib.crc32(output) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"Lossless CRC32 mismatch! Expected: {expected_crc:#x}, got: {actual_crc:#x}")

    return bytes(output)

# --- CLI & Testing ---
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python apex_compress.py -c <input_file> <output.apex>")
        print("  python apex_compress.py -d <input.apex> <output_file>")
        print("  python apex_compress.py -t <input_file>")
        sys.exit(1)

    mode = sys.argv[1]
    in_path = sys.argv[2]

    if mode == "-c":
        out_path = sys.argv[3]
        with open(in_path, 'rb') as f:
            raw = f.read()
        t0 = time.time()
        comp = compress(raw)
        dt = (time.time() - t0) * 1000.0
        with open(out_path, 'wb') as f:
            f.write(comp)
        ratio = (len(comp) / len(raw) * 100.0) if raw else 0.0
        bpc = (len(comp) * 8.0 / len(raw)) if raw else 0.0
        print(f"[ApexCompress] Compressed {len(raw)} -> {len(comp)} bytes ({ratio:.2f}% | {bpc:.3f} bpc) in {dt:.1f} ms")
        print(f"[ApexCompress] Saved to {out_path}")

    elif mode == "-d":
        out_path = sys.argv[3]
        with open(in_path, 'rb') as f:
            comp = f.read()
        t0 = time.time()
        decomp = decompress(comp)
        dt = (time.time() - t0) * 1000.0
        with open(out_path, 'wb') as f:
            f.write(decomp)
        print(f"[ApexCompress] Decompressed {len(comp)} -> {len(decomp)} bytes in {dt:.1f} ms")
        print(f"[ApexCompress] Verified bit-for-bit lossless integrity (CRC32 valid).")
        print(f"[ApexCompress] Restored to {out_path}")

    elif mode == "-t":
        with open(in_path, 'rb') as f:
            raw = f.read()
        t0 = time.time()
        comp = compress(raw)
        t1 = time.time()
        decomp = decompress(comp)
        t2 = time.time()

        c_ms = (t1 - t0) * 1000.0
        d_ms = (t2 - t1) * 1000.0
        ratio = (len(comp) / len(raw) * 100.0) if raw else 0.0
        bpc = (len(comp) * 8.0 / len(raw)) if raw else 0.0

        print("--------------------------------------------------------")
        print(f" Original Size   : {len(raw)} bytes")
        print(f" Compressed Size : {len(comp)} bytes")
        print(f" Compression Ratio: {ratio:.2f}% ({100.0 - ratio:.2f}% space saved)")
        print(f" Bits Per Character (bpc): {bpc:.3f} bpc")
        print(f" Compression Time: {c_ms:.1f} ms")
        print(f" Decompression Time: {d_ms:.1f} ms")
        if raw == decomp:
            print(" Integrity Check: PASS (100% bit-exact match, zero loss)")
        else:
            print(" Integrity Check: FAIL (Decompressed bytes differ!)")
            sys.exit(2)
        print("--------------------------------------------------------")

if __name__ == '__main__':
    main()
