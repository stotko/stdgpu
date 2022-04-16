#!/bin/bash
set -e

# Create external directory
cmake -E cmake_echo_color --blue ">>>>> Download external dependencies"
sh scripts/utils/create_empty_directory.sh external

# Download thrust
sh scripts/utils/download_thrust.sh
