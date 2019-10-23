/*
 *  Copyright 2019 Patrick Stotko
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

#ifndef STDGPU_PLATFORM_H
#define STDGPU_PLATFORM_H

/**
 * \file stdgpu/platform.h
 */



namespace stdgpu
{

/**
 * \brief Host compiler: Unknown
 */
#define STDGPU_HOST_COMPILER_UNKNOWN -1
/**
 * \brief Host compiler: GCC
 */
#define STDGPU_HOST_COMPILER_GCC      0
/**
 * \brief Host compiler: Clang
 */
#define STDGPU_HOST_COMPILER_CLANG    1
/**
 * \brief Host compiler: Microsoft Visual C++
 */
#define STDGPU_HOST_COMPILER_MSVC     2

/**
 * \brief Device compiler: Unknown
 */
#define STDGPU_DEVICE_COMPILER_UNKNOWN -1
/**
 * \brief Device compiler: NVCC
 */
#define STDGPU_DEVICE_COMPILER_NVCC     0

/**
 * \def STDGPU_HOST_COMPILER
 * \brief The host compiler
 */
#if defined(__GNUC__)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_GCC
#elif defined(__clang__)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_CLANG
#elif defined(_MSC_VER)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_MSVC
#else
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_UNKNOWN
#endif

/**
 * \def STDGPU_DEVICE_COMPILER
 * \brief The device compiler
 */
#if defined(__CUDACC__)
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_NVCC
#else
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_UNKNOWN
#endif


/**
 * \def STDGPU_HOST_DEVICE
 * \brief Platform-independent __host__ __device__ function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
    #define STDGPU_HOST_DEVICE __host__ __device__
#else
    #define STDGPU_HOST_DEVICE
#endif


/**
 * \def STDGPU_DEVICE_ONLY
 * \brief Platform-independent __device__ function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
    #define STDGPU_DEVICE_ONLY __device__
#else
    // Should trigger a compact error message containing the error string
    #define STDGPU_DEVICE_ONLY sizeof("STDGPU ERROR: Wrong compiler detected! Device-only functions must be compiled with the device compiler!")
#endif


/**
 * \def STDGPU_CONSTANT
 * \brief Platform-independent _constant__ variable annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
    #define STDGPU_CONSTANT __constant__
#else
    #define STDGPU_CONSTANT
#endif


/**
 * \brief Code path: Host
 */
#define STDGPU_CODE_HOST   0
/**
 * \brief Code path: Device
 */
#define STDGPU_CODE_DEVICE 1

/**
 * \def STDGPU_CODE
 * \brief The code path
 */
#if defined(__CUDA_ARCH__)
    #define STDGPU_CODE STDGPU_CODE_DEVICE
#else
    #define STDGPU_CODE STDGPU_CODE_HOST
#endif

} // namespace stdgpu



#endif // STDGPU_PLATFORM_H
