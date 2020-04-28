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

#include <stdgpu/contract.h>
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
        old = atomicCAS(address_as_int, assumed, __float_as_int( fminf(__int_as_float(assumed), value) ));

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
        old = atomicCAS(address_as_int, assumed, __float_as_int( fmaxf(__int_as_float(assumed), value) ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __int_as_float(old);
}


namespace stdgpu
{

namespace hip
{

template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address,
                const T desired)
{
    return atomicExch(address, desired);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address,
                        const T expected,
                        const T desired)
{
    return atomicCAS(address, expected, desired);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address,
                 const T arg)
{
    return atomicAdd(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address,
                 const T arg)
{
    return atomicSub(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address,
                 const T arg)
{
    return atomicAnd(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address,
                 const T arg)
{
    return atomicOr(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address,
                 const T arg)
{
    return atomicXor(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address,
                 const T arg)
{
    return atomicMin(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address,
                 const T arg)
{
    return atomicMax(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address,
                     const T arg)
{
    return atomicInc(address, arg);
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address,
                     const T arg)
{
    return atomicDec(address, arg);
}

} // namespace hip

} // namespace stdgpu



#endif // STDGPU_HIP_ATOMIC_DETAIL_H
