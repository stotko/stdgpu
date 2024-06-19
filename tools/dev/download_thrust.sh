#!/bin/bash
set -e

cmake -E cmake_echo_color --blue ">>>>> Download thrust"
cmake -E chdir external git clone https://github.com/NVIDIA/cccl
cmake -E chdir external/cccl git fetch --all --tags --prune
cmake -E chdir external/cccl git checkout tags/v2.2.0
