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

#ifndef STDGPU_HIP_EXECUTION_H
#define STDGPU_HIP_EXECUTION_H

#include <stdgpu/platform.h>

namespace stdgpu::hip
{

/**
 * \brief Runs a (spinlock-based) critical section with one wavefront lane active at a time
 * \param[in] body The callable to execute
 *
 * AMD GPUs (CDNA wave64, RDNA wave32) provide no per-lane forward-progress
 * guarantee, so a lane that acquires a lock cannot advance past reconvergence
 * while its peers keep spinning -- a livelock. Electing one active lane at a
 * time from the active-lane mask lets each lane run the body in turn.
 */
template <typename F>
STDGPU_DEVICE_ONLY void
warp_convergent_execute(F&& body)
{
    // Lane within the wavefront, expressed with the documented threadIdx/warpSize
    // builtins: HIP numbers threads within a wavefront in row-major (x fastest)
    // order, so the lane is the linear thread index modulo the wavefront size.
    const unsigned int linear_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(linear_thread_id % static_cast<unsigned int>(warpSize));
    unsigned long long active = __activemask();

    while (active)
    {
        int elected = __ffsll(static_cast<long long>(active)) - 1;
        if (lane == elected)
        {
            body();
        }
        active &= ~(1ull << elected);
    }
}

} // namespace stdgpu::hip

#endif // STDGPU_HIP_EXECUTION_H
