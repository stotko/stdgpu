#!/bin/bash
set -e

# Verify headers
cmake -E cmake_echo_color --blue ">>>>> Verify headers"
cmake --build build --target stdgpu_verify_interface_header_sets
