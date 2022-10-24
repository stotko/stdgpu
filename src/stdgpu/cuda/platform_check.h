/*
 *  Copyright 2021 Patrick Stotko
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

#ifndef STDGPU_CUDA_PLATFORM_CHECK_H
#define STDGPU_CUDA_PLATFORM_CHECK_H

#include <stdgpu/compiler.h>

namespace stdgpu::cuda
{

#if STDGPU_DEVICE_COMPILER != STDGPU_DEVICE_COMPILER_NVCC && STDGPU_DEVICE_COMPILER != STDGPU_DEVICE_COMPILER_CUDACLANG
    #error STDGPU ERROR: Wrong compiler detected! You are including a file with functions that must be compiled with the device compiler!
    #include <stdgpu/this_include_error_is_intended/take_a_look_at_the_error_above> // Helper to stop compilation as early as possible
#endif

} // namespace stdgpu::cuda

#endif // STDGPU_CUDA_PLATFORM_CHECK_H
