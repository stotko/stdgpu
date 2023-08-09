#!/bin/bash
set -e

cmake -E cmake_echo_color --blue ">>>>> Download thrust"
cmake -E chdir external git clone https://github.com/NVIDIA/thrust
cmake -E chdir external/thrust git fetch --all --tags --prune
cmake -E chdir external/thrust git checkout tags/1.17.2
