#!/bin/bash
set -e

# Apply code style
cmake -E cmake_echo_color --blue ">>>>> Apply code style"
cmake --build build --target apply_code_style
