#!/bin/sh
set -e

cmake -E cmake_echo_color --blue ">>>>> Download thrust"
cmake -E chdir external git clone https://github.com/NVIDIA/thrust
