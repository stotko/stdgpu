#!/bin/sh
set -e

# GCC: Already installed
# Clang: Install libomp-dev and remove conflicting packages
sudo apt-get update
sudo apt-get remove libomp*
sudo apt-get install -f libomp-dev
