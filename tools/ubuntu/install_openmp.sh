#!/bin/bash
set -e

# GCC: Already installed
# Clang: Install libomp-dev and remove conflicting packages
sudo apt-get update
sudo apt-get install libomp-dev
