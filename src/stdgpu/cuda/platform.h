/*
 *  Copyright 2020 Patrick Stotko
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef STDGPU_CUDA_PLATFORM_H
#define STDGPU_CUDA_PLATFORM_H

#include <stdgpu/compiler.h>

namespace stdgpu::cuda
{

/**
 * \def STDGPU_CUDA_HOST_DEVICE
 * \brief Platform-independent host device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_CUDACLANG
    #define STDGPU_CUDA_HOST_DEVICE __host__ __device__
#else
    #define STDGPU_CUDA_HOST_DEVICE
#endif

/**
 * \def STDGPU_CUDA_DEVICE_ONLY
 * \brief Platform-independent device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_CUDACLANG
    #define STDGPU_CUDA_DEVICE_ONLY __device__
#else
    // Undefined
#endif

/**
 * \def STDGPU_CUDA_CONSTANT
 * \brief Platform-independent constant variable annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_CUDACLANG
    #define STDGPU_CUDA_CONSTANT __constant__
#else
    #define STDGPU_CUDA_CONSTANT
#endif

/**
 * \def STDGPU_CUDA_IS_DEVICE_CODE
 * \brief Platform-independent device code detection
 */
#if defined(__CUDA_ARCH__)
    #define STDGPU_CUDA_IS_DEVICE_CODE 1
#else
    #define STDGPU_CUDA_IS_DEVICE_CODE 0
#endif

/**
 * \def STDGPU_CUDA_IS_DEVICE_COMPILED
 * \brief Platform-independent device compilation detection
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_CUDACLANG
    #define STDGPU_CUDA_IS_DEVICE_COMPILED 1
#else
    #define STDGPU_CUDA_IS_DEVICE_COMPILED 0
#endif

} // namespace stdgpu::cuda

#endif // STDGPU_CUDA_PLATFORM_H
