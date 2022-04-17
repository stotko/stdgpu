cmake_minimum_required(VERSION 3.15)

project(install_test LANGUAGES CXX)


find_package(stdgpu REQUIRED)


add_executable(install_test)

target_sources(install_test PRIVATE install_test.cpp)

target_link_libraries(install_test PRIVATE stdgpu::stdgpu)

