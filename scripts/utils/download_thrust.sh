#!/bin/sh
set -e

cmake -E cmake_echo_color --blue ">>>>> Download thrust"
cmake -E chdir external git clone https://github.com/thrust/thrust

# Latest release version
cmake -E chdir external/thrust git checkout 1.9.10
