/*
 *  Copyright 2024 Patrick Stotko
 *  Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef STDGPU_CUDA_EXECUTION_H
#define STDGPU_CUDA_EXECUTION_H

#include <stdgpu/platform.h>

namespace stdgpu::cuda
{

/**
 * \brief Runs a (spinlock-based) critical section
 * \param[in] body The callable to execute
 *
 * NVIDIA Volta and later provide Independent Thread Scheduling, so a lane that
 * acquires a lock makes forward progress while its peers spin; no per-lane
 * serialization is required.
 */
template <typename F>
STDGPU_DEVICE_ONLY void
warp_convergent_execute(F&& body)
{
    body();
}

} // namespace stdgpu::cuda

#endif // STDGPU_CUDA_EXECUTION_H
