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

#ifndef STDGPU_COMPILER_H
#define STDGPU_COMPILER_H

/**
 * \addtogroup compiler compiler
 * \ingroup system
 * @{
 */

/**
 * \file stdgpu/compiler.h
 */

namespace stdgpu
{

/**
 * \ingroup compiler
 * \brief Host compiler: Unknown
 */
#define STDGPU_HOST_COMPILER_UNKNOWN 10
/**
 * \ingroup compiler
 * \brief Host compiler: GCC
 */
#define STDGPU_HOST_COMPILER_GCC 11
/**
 * \ingroup compiler
 * \brief Host compiler: Clang
 */
#define STDGPU_HOST_COMPILER_CLANG 12
/**
 * \ingroup compiler
 * \brief Host compiler: Microsoft Visual C++
 */
#define STDGPU_HOST_COMPILER_MSVC 13

/**
 * \ingroup compiler
 * \brief Device compiler: Unknown
 */
#define STDGPU_DEVICE_COMPILER_UNKNOWN 20
/**
 * \ingroup compiler
 * \brief Device compiler: NVCC
 */
#define STDGPU_DEVICE_COMPILER_NVCC 21
/**
 * \ingroup compiler
 * \brief Device compiler: HIP-Clang
 */
#define STDGPU_DEVICE_COMPILER_HIPCLANG 22
/**
 * \ingroup compiler
 * \brief Device compiler: CUDA-Clang
 */
#define STDGPU_DEVICE_COMPILER_CUDACLANG 23

/**
 * \ingroup compiler
 * \def STDGPU_HOST_COMPILER
 * \brief The detected host compiler
 */
#if defined(__GNUC__) && !defined(__clang__)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_GCC
#elif defined(__clang__)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_CLANG
#elif defined(_MSC_VER)
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_MSVC
#else
    #define STDGPU_HOST_COMPILER STDGPU_HOST_COMPILER_UNKNOWN
#endif

/**
 * \ingroup compiler
 * \def STDGPU_DEVICE_COMPILER
 * \brief The detected device compiler
 */
#if defined(__NVCC__)
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_NVCC
#elif defined(__HIP__)
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_HIPCLANG
#elif defined(__clang__) && defined(__CUDA__)
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_CUDACLANG
#else
    #define STDGPU_DEVICE_COMPILER STDGPU_DEVICE_COMPILER_UNKNOWN
#endif

} // namespace stdgpu

/**
 * @}
 */

#endif // STDGPU_COMPILER_H
