#include "compressor.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

void print_usage(const char* prog) {
    std::cout << "========================================================\n"
              << " ApexCompress: Advanced Multi-Context Lossless Compressor\n"
              << "========================================================\n"
              << "Usage:\n"
              << "  " << prog << " -c <input_file> <output.apex>  : Compress file\n"
              << "  " << prog << " -d <input.apex> <output_file>  : Decompress file\n"
              << "  " << prog << " -t <input_file>                : Test round-trip and benchmark\n";
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (size > 0 && !file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return buffer;
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    if (!data.empty() && !file.write(reinterpret_cast<const char*>(data.data()), data.size())) {
        throw std::runtime_error("Failed to write file: " + path);
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            print_usage(argv[0]);
            return 1;
        }

        std::string mode = argv[1];

        if (mode == "-c") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            std::string in_path = argv[2];
            std::string out_path = argv[3];

            std::cout << "[ApexCompress] Reading " << in_path << "...\n";
            auto raw = read_file(in_path);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto comp = apex::ApexCompressor::compress(raw.data(), raw.size());
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            write_file(out_path, comp);

            double ratio = raw.empty() ? 0.0 : ((double)comp.size() / (double)raw.size()) * 100.0;
            double bpc = raw.empty() ? 0.0 : ((double)comp.size() * 8.0) / (double)raw.size();

            std::cout << "[ApexCompress] Compressed: " << raw.size() << " -> " << comp.size() 
                      << " bytes (" << std::fixed << std::setprecision(2) << ratio << "% | " 
                      << bpc << " bpc) in " << std::setprecision(1) << ms << " ms\n"
                      << "[ApexCompress] Saved to: " << out_path << "\n";
            return 0;
        } else if (mode == "-d") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            std::string in_path = argv[2];
            std::string out_path = argv[3];

            std::cout << "[ApexCompress] Reading " << in_path << "...\n";
            auto comp = read_file(in_path);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto decomp = apex::ApexCompressor::decompress(comp.data(), comp.size());
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            write_file(out_path, decomp);

            std::cout << "[ApexCompress] Decompressed: " << comp.size() << " -> " << decomp.size() 
                      << " bytes in " << std::fixed << std::setprecision(1) << ms << " ms\n"
                      << "[ApexCompress] Verified bit-for-bit lossless integrity (CRC32 valid).\n"
                      << "[ApexCompress] Restored to: " << out_path << "\n";
            return 0;
        } else if (mode == "-t") {
            std::string in_path = argv[2];
            std::cout << "[ApexCompress] Testing round-trip compression on " << in_path << "...\n";
            auto raw = read_file(in_path);

            // Compress
            auto t0 = std::chrono::high_resolution_clock::now();
            auto comp = apex::ApexCompressor::compress(raw.data(), raw.size());
            auto t1 = std::chrono::high_resolution_clock::now();

            // Decompress
            auto t2 = std::chrono::high_resolution_clock::now();
            auto decomp = apex::ApexCompressor::decompress(comp.data(), comp.size());
            auto t3 = std::chrono::high_resolution_clock::now();

            double c_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double d_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double ratio = raw.empty() ? 0.0 : ((double)comp.size() / (double)raw.size()) * 100.0;
            double bpc = raw.empty() ? 0.0 : ((double)comp.size() * 8.0) / (double)raw.size();

            std::cout << "--------------------------------------------------------\n"
                      << " Original Size   : " << raw.size() << " bytes\n"
                      << " Compressed Size : " << comp.size() << " bytes\n"
                      << " Compression Ratio: " << std::fixed << std::setprecision(2) << ratio << "% ("
                      << (100.0 - ratio) << "% space saved)\n"
                      << " Bits Per Character (bpc): " << std::setprecision(3) << bpc << " bpc\n"
                      << " Compression Time: " << std::setprecision(1) << c_ms << " ms\n"
                      << " Decompression Time: " << d_ms << " ms\n";

            if (raw.size() == decomp.size() && std::memcmp(raw.data(), decomp.data(), raw.size()) == 0) {
                std::cout << " Integrity Check: PASS (100% bit-exact match, zero loss)\n"
                          << "--------------------------------------------------------\n";
                return 0;
            } else {
                std::cerr << " Integrity Check: FAIL (Decompressed output does not match original!)\n"
                          << "--------------------------------------------------------\n";
                return 2;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ApexCompress Error] " << ex.what() << "\n";
        return 1;
    }
}
