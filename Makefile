CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Isrc -march=native
SRC = src/main.cpp
BIN = bin/apex

all: $(BIN)

$(BIN): $(SRC) src/*.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

test: $(BIN)
	python tests/generate_test_data.py
	python apex_compress.py -t benchmark_data/sample_patterns.bin
	python apex_compress.py -t benchmark_data/sample_code.json
	python apex_compress.py -t benchmark_data/sample_text.txt
	@echo "All tests passed!"

benchmark:
	python tests/benchmark_comparison.py

clean:
	rm -rf bin build *.apex

.PHONY: all test benchmark clean
