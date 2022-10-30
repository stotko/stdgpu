/*
 *  Copyright 2019 Patrick Stotko
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

#ifndef STDGPU_OPENMP_ATOMIC_DETAIL_H
#define STDGPU_OPENMP_ATOMIC_DETAIL_H

#include <algorithm>

#include <stdgpu/platform.h>

namespace stdgpu::openmp
{

inline STDGPU_HOST_DEVICE bool
atomic_is_lock_free() noexcept
{
    return false;
}

inline STDGPU_DEVICE_ONLY void
atomic_thread_fence() noexcept
{
#pragma omp flush
}

template <typename T>
STDGPU_DEVICE_ONLY T
atomic_load(T* address) noexcept
{
    T current;
#pragma omp critical
    current = *address;
    return current;
}

template <typename T>
STDGPU_DEVICE_ONLY void
atomic_store(T* address, const T desired) noexcept
{
#pragma omp critical
    *address = desired;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address, const T desired) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = desired;
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address, const T expected, const T desired) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = (old == expected) ? desired : old;
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address, const T arg) noexcept
{
    T old;
#if STDGPU_OPENMP_DETAIL_VERSION >= STDGPU_OPENMP_DETAIL_VERSION_3_1
    #pragma omp atomic capture
#else
    #pragma omp critical
#endif
    {
        old = *address;
        *address += arg;
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address, const T arg) noexcept
{
    T old;
#if STDGPU_OPENMP_DETAIL_VERSION >= STDGPU_OPENMP_DETAIL_VERSION_3_1
    #pragma omp atomic capture
#else
    #pragma omp critical
#endif
    {
        old = *address;
        *address -= arg;
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address, const T arg) noexcept
{
    T old;
#if STDGPU_OPENMP_DETAIL_VERSION >= STDGPU_OPENMP_DETAIL_VERSION_3_1
    #pragma omp atomic capture
#else
    #pragma omp critical
#endif
    {
        old = *address;
        *address &= arg; // NOLINT(hicpp-signed-bitwise)
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address, const T arg) noexcept
{
    T old;
#if STDGPU_OPENMP_DETAIL_VERSION >= STDGPU_OPENMP_DETAIL_VERSION_3_1
    #pragma omp atomic capture
#else
    #pragma omp critical
#endif
    {
        old = *address;
        *address |= arg; // NOLINT(hicpp-signed-bitwise)
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address, const T arg) noexcept
{
    T old;
#if STDGPU_OPENMP_DETAIL_VERSION >= STDGPU_OPENMP_DETAIL_VERSION_3_1
    #pragma omp atomic capture
#else
    #pragma omp critical
#endif
    {
        old = *address;
        *address ^= arg; // NOLINT(hicpp-signed-bitwise)
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address, const T arg) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = std::min(old, arg);
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address, const T arg) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = std::max(old, arg);
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same_v<T, unsigned int>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address, const T arg) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = (old >= arg) ? T(0) : old + T(1);
    }
    return old;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same_v<T, unsigned int>)>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address, const T arg) noexcept
{
    T old;
#pragma omp critical
    {
        old = *address;
        *address = (old == T(0) || old > arg) ? arg : old - T(1);
    }
    return old;
}

} // namespace stdgpu::openmp

#endif // STDGPU_OPENMP_ATOMIC_DETAIL_H
