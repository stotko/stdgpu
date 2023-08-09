#!/bin/bash
set -e

# Install docs dependencies
sudo apt-get update
sudo apt-get install bison flex python3
pip install -r docs/requirements.txt
