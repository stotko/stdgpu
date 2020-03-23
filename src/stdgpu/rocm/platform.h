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

#ifndef STDGPU_ROCM_PLATFORM_H
#define STDGPU_ROCM_PLATFORM_H


// HCC should automatically include its runtime defines similar to NVCC
#include <hip/hip_runtime.h>

#include <stdgpu/compiler.h>



namespace stdgpu
{
namespace rocm
{

/**
 * \def STDGPU_ROCM_HOST_DEVICE
 * \hideinitializer
 * \brief Platform-independent host device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HCC
    #define STDGPU_ROCM_HOST_DEVICE __host__ __device__
#else
    #define STDGPU_ROCM_HOST_DEVICE
#endif


/**
 * \def STDGPU_ROCM_DEVICE_ONLY
 * \hideinitializer
 * \brief Platform-independent device function annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HCC
    #define STDGPU_ROCM_DEVICE_ONLY __device__
#else
    // Should trigger a compact error message containing the error string
    #define STDGPU_ROCM_DEVICE_ONLY sizeof("STDGPU ERROR: Wrong compiler detected! Device-only functions must be compiled with the device compiler!")
#endif


/**
 * \def STDGPU_ROCM_CONSTANT
 * \hideinitializer
 * \brief Platform-independent constant variable annotation
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HCC
    #define STDGPU_ROCM_CONSTANT __constant__
#else
    #define STDGPU_ROCM_CONSTANT
#endif


/**
 * \def STDGPU_ROCM_IS_DEVICE_CODE
 * \hideinitializer
 * \brief Platform-independent device code detection
 */
#if defined(__HIP_DEVICE_COMPILE__)
    #define STDGPU_ROCM_IS_DEVICE_CODE 1
#else
    #define STDGPU_ROCM_IS_DEVICE_CODE 0
#endif


/**
 * \def STDGPU_ROCM_IS_DEVICE_COMPILED
 * \hideinitializer
 * \brief Platform-independent device compilation detection
 */
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_HCC
    #define STDGPU_ROCM_IS_DEVICE_COMPILED 1
#else
    #define STDGPU_ROCM_IS_DEVICE_COMPILED 0
#endif


} // namespace rocm

} // namespace stdgpu



#endif // STDGPU_ROCM_PLATFORM_H
