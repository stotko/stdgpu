#!/bin/bash
set -e

cmake -E cmake_echo_color --blue ">>>>> Download doxygen"
cmake -E chdir external git clone https://github.com/doxygen/doxygen
cmake -E chdir external/doxygen git fetch --all --tags --prune
cmake -E chdir external/doxygen git checkout tags/Release_1_9_1

cmake -E cmake_echo_color --blue ">>>>> Build doxygen"
cmake -E chdir external/doxygen cmake -E make_directory build
cmake -E chdir external/doxygen cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=bin
cmake -E chdir external/doxygen cmake --build build --config Release --parallel 13
cmake -E chdir external/doxygen cmake --build build --target install
