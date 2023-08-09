#!/bin/bash
set -e

# Create external directory
cmake -E cmake_echo_color --blue ">>>>> Download external dependencies"
sh tools/backend/helper/create_empty_directory.sh external

# Download thrust
sh tools/dev/download_thrust.sh
