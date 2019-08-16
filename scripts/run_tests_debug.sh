#!/bin/sh

# Run tests
cmake -E chdir build ctest -V -C Debug
