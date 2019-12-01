#!/bin/sh
set -e

# Check number of input parameters
if [ "$#" -ne 2 ]; then
    cmake -E echo "set_cxx_compiler_ubuntu: Expected 2 parameters, but received $# parameters"
    exit 1
fi

# Set C compiler
sudo update-alternatives --install /usr/bin/cc cc /usr/bin/$1 100
sudo update-alternatives --set cc /usr/bin/$1

# Set C++ compiler
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/$2 100
sudo update-alternatives --set c++ /usr/bin/$2
