# Contributing to ApexCompress

Thank you for your interest in improving ApexCompress!

ApexCompress is an open-source project exploring state-of-the-art lossless data compression through multi-context modeling, neural probability estimation, and adaptive arithmetic range coding.

---

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhil78957/ApexCompress.git
   cd ApexCompress
   ```

2. Run test suite:
   ```bash
   python tests/generate_test_data.py
   python tests/benchmark_comparison.py
   ```

3. Build C++ binary:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```

---

## Areas for Exploration

We welcome contributions in the following areas:
- **New Context Models**: Word-ngram models, record-stride detectors for multi-channel audio or image bitmaps, and LZP (Lempel-Ziv-Poisson) predictors.
- **Hardware Acceleration**: AVX2 / AVX-512 SIMD vectorization for the dot-product inner loop in `mixer.hpp`.
- **Secondary Symbol Estimation**: Adaptive quantization bin boundaries based on empirical variance.
- **Multi-threaded Block Compression**: Chunking large files into independent streams with parallel thread execution.

---

## Pull Request Guidelines

1. Ensure 100% exact bit-for-bit lossless roundtrip on all benchmark test files (`assert dec == raw`).
2. Adhere to modern C++20 conventions (`std::span`, `constexpr`, zero undefined behavior).
3. Update or add unit tests for any new context models or transformations.
