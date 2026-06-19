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

#ifndef STDGPU_WAVE_LOCK_H
#define STDGPU_WAVE_LOCK_H

#include <stdgpu/platform.h>

namespace stdgpu::detail
{

/**
 * Wave-serialized locking helper for AMD CDNA/RDNA GPUs.
 *
 * On CUDA Volta+, Independent Thread Scheduling allows a lane that wins
 * a spinlock to make forward progress while other lanes spin. On AMD GPUs
 * (CDNA wave64, RDNA wave32), there is no per-lane forward-progress
 * guarantee: a lane that wins a CAS is stuck at SIMT reconvergence waiting
 * on losers who spin forever -- classic livelock.
 *
 * The fix: wave-serialize by electing one active lane at a time via ballot,
 * letting only that lane attempt the lock. Other lanes wait their turn.
 *
 * Usage:
 *   wave_lock_serialize([&]() {
 *       // Your spinlock-based critical section
 *       while (!done) {
 *           if (lock.try_lock()) {
 *               // critical section
 *               done = true;
 *               lock.unlock();
 *           }
 *       }
 *   });
 */

#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)

// AMD HIP: wave-serialize using ballot to avoid wave64/wave32 livelock
template <typename F>
STDGPU_DEVICE_ONLY void
wave_lock_serialize(F&& body)
{
    int lane = static_cast<int>(__lane_id());
    unsigned long long active = __ballot(1);

    while (active)
    {
        // Elect the lowest-numbered active lane
        int elected = __ffsll(static_cast<long long>(active)) - 1;
        if (lane == elected)
        {
            body();
        }
        // Clear this lane's bit and move to the next
        active &= ~(1ull << elected);
    }
}

#else

// CUDA or other: Independent Thread Scheduling handles this, no serialization needed
template <typename F>
STDGPU_DEVICE_ONLY void
wave_lock_serialize(F&& body)
{
    body();
}

#endif

} // namespace stdgpu::detail

#endif // STDGPU_WAVE_LOCK_H
