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

#ifndef STDGPU_HIP_ATOMIC_DETAIL_H
#define STDGPU_HIP_ATOMIC_DETAIL_H

#include <stdgpu/algorithm.h>
#include <stdgpu/limits.h>
#include <stdgpu/platform.h>



inline STDGPU_DEVICE_ONLY unsigned long long int
atomicSub(unsigned long long int* address,
          const unsigned long long int value)
{
    return atomicAdd(address, stdgpu::numeric_limits<unsigned long long int>::max() - value + 1);
}


inline STDGPU_DEVICE_ONLY float
atomicSub(float* address,
          const float value)
{
    return atomicAdd(address, -value);
}


inline STDGPU_DEVICE_ONLY float
atomicMin(float* address,
          const float value)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( stdgpu::min<float>(__int_as_float(assumed), value) ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __int_as_float(old);
}


inline STDGPU_DEVICE_ONLY float
atomicMax(float* address,
          const float value)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( stdgpu::max<float>(__int_as_float(assumed), value) ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __int_as_float(old);
}


namespace stdgpu
{

namespace hip
{

template <typename T>
STDGPU_DEVICE_ONLY T
atomic_load(T* address)
{
    __threadfence();

    volatile T* volatile_address = address;
    T current = *volatile_address;

    __threadfence();

    return current;
}


template <typename T>
STDGPU_DEVICE_ONLY void
atomic_store(T* address,
             const T desired)
{
    __threadfence();

    volatile T* volatile_address = address;
    *volatile_address = desired;

    __threadfence();
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address,
                const T desired)
{
    __threadfence();

    T old = atomicExch(address, desired);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address,
                        const T expected,
                        const T desired)
{
    __threadfence();

    T old = atomicCAS(address, expected, desired);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicAdd(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicSub(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicAnd(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicOr(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicXor(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicMin(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address,
                 const T arg)
{
    __threadfence();

    T old = atomicMax(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address,
                     const T arg)
{
    __threadfence();

    T old = atomicInc(address, arg);

    __threadfence();

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address,
                     const T arg)
{
    __threadfence();

    T old = atomicDec(address, arg);

    __threadfence();

    return old;
}

} // namespace hip

} // namespace stdgpu



#endif // STDGPU_HIP_ATOMIC_DETAIL_H
