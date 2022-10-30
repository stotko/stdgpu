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

namespace stdgpu::hip
{

inline STDGPU_HOST_DEVICE bool
atomic_is_lock_free() noexcept
{
    return true;
}

inline STDGPU_DEVICE_ONLY void
atomic_thread_fence() noexcept
{
    __threadfence();
}

template <typename T>
STDGPU_DEVICE_ONLY T
atomic_load(T* address) noexcept
{
    volatile T* volatile_address = address;
    T current = *volatile_address;

    return current;
}

template <typename T>
STDGPU_DEVICE_ONLY void
atomic_store(T* address, const T desired) noexcept
{
    volatile T* volatile_address = address;
    *volatile_address = desired;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address, const T desired) noexcept
{
    return atomicExch(address, desired);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address, const T expected, const T desired) noexcept
{
    return atomicCAS(address, expected, desired);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address, const T arg) noexcept
{
    return atomicAdd(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address, const T arg) noexcept
{
    return atomicSub(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address, const T arg) noexcept
{
    return atomicAnd(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address, const T arg) noexcept
{
    return atomicOr(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address, const T arg) noexcept
{
    return atomicXor(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address, const T arg) noexcept
{
    return atomicMin(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address, const T arg) noexcept
{
    return atomicMax(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same_v<T, unsigned int>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address, const T arg) noexcept
{
    return atomicInc(address, arg);
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same_v<T, unsigned int>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address, const T arg) noexcept
{
    return atomicDec(address, arg);
}

} // namespace stdgpu::hip

#endif // STDGPU_HIP_ATOMIC_DETAIL_H
