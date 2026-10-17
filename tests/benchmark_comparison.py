"""
Benchmark comparison:
ApexCompress vs Gzip (Deflate) vs Bzip2 (BWT) vs LZMA (7z)
"""
import os
import sys
import gzip
import bz2
import lzma
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apex_compress import compress, decompress

files = [
    ('Patterns & Repetition', 'benchmark_data/sample_patterns.bin'),
    ('Structured JSON Code', 'benchmark_data/sample_code.json'),
    ('English Literature', 'benchmark_data/sample_text.txt'),
]

print(f"{'Data Type':<25} | {'Original':<9} | {'Gzip (lvl 9)':<12} | {'Bzip2 (lvl 9)':<13} | {'LZMA (lvl 9)':<12} | {'ApexCompress':<14} | {'Apex Gain vs LZMA':<17}")
print("-" * 115)

for title, path in files:
    with open(path, 'rb') as f:
        raw = f.read()
    orig_len = len(raw)

    gz_data = gzip.compress(raw, compresslevel=9)
    bz_data = bz2.compress(raw, compresslevel=9)
    lz_data = lzma.compress(raw, preset=9)
    
    t0 = time.time()
    apex_data = compress(raw)
    t1 = time.time()
    
    # Verify lossless integrity
    dec = decompress(apex_data)
    assert dec == raw, "Integrity failure!"

    gain = ((len(lz_data) - len(apex_data)) / len(lz_data)) * 100.0
    gain_str = f"+{gain:.1f}% smaller" if gain > 0 else f"{gain:.1f}%"

    print(f"{title:<25} | {orig_len:<9} | {len(gz_data):<12} | {len(bz_data):<13} | {len(lz_data):<12} | {len(apex_data):<14} | {gain_str:<17}")

print("-" * 115)
