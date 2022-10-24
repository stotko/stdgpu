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

#ifndef STDGPU_HIP_PLATFORM_H
#define STDGPU_HIP_PLATFORM_H

#include <stdgpu/compiler.h>

namespace stdgpu::hip
{

/**
 * \def STDGPU_HIP_HOST_DEVICE
 * \brief Platform-independent host device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HIPCLANG
    #define STDGPU_HIP_HOST_DEVICE __host__ __device__
#else
    #define STDGPU_HIP_HOST_DEVICE
#endif

/**
 * \def STDGPU_HIP_DEVICE_ONLY
 * \brief Platform-independent device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HIPCLANG
    #define STDGPU_HIP_DEVICE_ONLY __device__
#else
    // Undefined
#endif

/**
 * \def STDGPU_HIP_CONSTANT
 * \brief Platform-independent constant variable annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HIPCLANG
    #define STDGPU_HIP_CONSTANT __constant__
#else
    #define STDGPU_HIP_CONSTANT
#endif

/**
 * \def STDGPU_HIP_IS_DEVICE_CODE
 * \brief Platform-independent device code detection
 */
#if defined(__HIP_DEVICE_COMPILE__)
    #define STDGPU_HIP_IS_DEVICE_CODE 1
#else
    #define STDGPU_HIP_IS_DEVICE_CODE 0
#endif

/**
 * \def STDGPU_HIP_IS_DEVICE_COMPILED
 * \brief Platform-independent device compilation detection
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HIPCLANG
    #define STDGPU_HIP_IS_DEVICE_COMPILED 1
#else
    #define STDGPU_HIP_IS_DEVICE_COMPILED 0
#endif

} // namespace stdgpu::hip

#endif // STDGPU_HIP_PLATFORM_H
