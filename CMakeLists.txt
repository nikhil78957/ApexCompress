cmake_minimum_required(VERSION 3.16)
project(ApexCompress
    VERSION 1.0.0
    DESCRIPTION "State-of-the-Art Multi-Context Neural Lossless Compressor"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization compiler flags
if(MSVC)
    add_compile_options(/O2 /W4 /permissive-)
else()
    add_compile_options(-O3 -Wall -Wextra -pedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-march=native)
    endif()
endif()

include_directories(src)

# Core Apex CLI Executable
add_executable(apex
    src/main.cpp
    src/common.hpp
    src/range_coder.hpp
    src/state_table.hpp
    src/context_model.hpp
    src/match_model.hpp
    src/text_model.hpp
    src/mixer.hpp
    src/sse.hpp
    src/compressor.hpp
)

set_target_properties(apex PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS apex DESTINATION bin)
