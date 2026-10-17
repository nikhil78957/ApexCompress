"""
Generate a diverse benchmark corpus for testing lossless compression:
1. English text & literature
2. Source code (Python / C / JSON)
3. Structured tabular / columnar numbers
4. Repetitive patterns & runs
5. Semi-random binary executable simulation
"""

import os

os.makedirs('benchmark_data', exist_ok=True)

# 1. Text sample: Alice in Wonderland excerpt
text_sample = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: 
once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 
'and what is the use of a book,' thought Alice 'without pictures or conversations?'

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), 
whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, 
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit 
say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she 
ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch 
out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across 
her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and 
burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large 
rabbit-hole under the hedge.
""" * 4

with open('benchmark_data/sample_text.txt', 'wb') as f:
    f.write(text_sample.encode('utf-8'))

# 2. Source code / JSON structured data
json_sample = """
{
  "system": "ApexCompress Lossless Engine",
  "version": "1.0.0",
  "benchmarks": [
    {"name": "order0", "type": "intra_byte", "weights": [1.0, 0.5, 0.25], "active": true},
    {"name": "order1", "type": "markov", "weights": [1.2, 0.8, 0.45], "active": true},
    {"name": "order2", "type": "context_hash", "weights": [2.1, 1.4, 0.95], "active": true},
    {"name": "match_model", "type": "lz_string", "weights": [3.5, 2.8, 1.95], "active": true},
    {"name": "text_model", "type": "word_grammar", "weights": [1.8, 1.2, 0.85], "active": true}
  ],
  "algorithms_evaluated": [
    {"algo": "Deflate", "family": "LZ77+Huffman", "entropy_limit_gap": "wide"},
    {"algo": "Bzip2", "family": "BWT+MTF+Huffman", "entropy_limit_gap": "moderate"},
    {"algo": "LZMA", "family": "LZ77+RangeCoder", "entropy_limit_gap": "narrow"},
    {"algo": "PAQ/Apex", "family": "MultiContext+Neural+Arithmetic", "entropy_limit_gap": "minimal"}
  ]
}
""" * 5

with open('benchmark_data/sample_code.json', 'wb') as f:
    f.write(json_sample.encode('utf-8'))

# 3. Repeated pattern data
rep_sample = (b"ApexCompress" * 100) + (b"\x00" * 300) + (b"Header\tData\tFooter\n" * 50)
with open('benchmark_data/sample_patterns.bin', 'wb') as f:
    f.write(rep_sample)

print("Generated benchmark files:")
for fname in os.listdir('benchmark_data'):
    p = os.path.join('benchmark_data', fname)
    print(f"  {fname}: {os.path.getsize(p)} bytes")
