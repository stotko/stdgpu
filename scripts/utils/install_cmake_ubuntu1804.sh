#!/bin/sh
set -e

# Install CMake 3.15+ from official Kitware repository (see https://apt.kitware.com/)
sudo apt-get update
sudo rm /usr/local/bin/ccmake* /usr/local/bin/cmake* /usr/local/bin/cpack* /usr/local/bin/ctest*
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
sudo apt-get install cmake
