/*
 *  Copyright 2019 Patrick Stotko
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

#ifndef STDGPU_EXECUTION_DETAIL_H
#define STDGPU_EXECUTION_DETAIL_H

#include <stdgpu/platform.h>
#include <stdgpu/utility.h>

#if STDGPU_BACKEND == STDGPU_BACKEND_CUDA
    #include STDGPU_DETAIL_BACKEND_HEADER(execution.cuh)
#else
    #include STDGPU_DETAIL_BACKEND_HEADER(execution.h)
#endif

namespace stdgpu::detail
{

/**
 * \brief Runs a spinlock-based critical section in a way that is safe on the active backend
 * \param[in] body The callable to execute
 */
template <typename F>
STDGPU_DEVICE_ONLY void
warp_convergent_execute(F&& body)
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::warp_convergent_execute(stdgpu::forward<F>(body));
}

} // namespace stdgpu::detail

#endif // STDGPU_EXECUTION_DETAIL_H
