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



namespace stdgpu
{

namespace openmp
{

template <typename T>
STDGPU_DEVICE_ONLY T
atomic_load(T* address)
{
    // Implicit flush due to omp critical

    T current;
    #pragma omp critical
    {
        current = *address;
    }

    // Implicit flush due to omp critical

    return current;
}


template <typename T>
STDGPU_DEVICE_ONLY void
atomic_store(T* address,
             const T desired)
{
    // Implicit flush due to omp critical

    #pragma omp critical
    {
        *address = desired;
    }

    // Implicit flush due to omp critical
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address,
                const T desired)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = desired;
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address,
                        const T expected,
                        const T desired)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = (old == expected) ? desired : old;
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = old + arg;
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = old - arg;
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = old & arg; // NOLINT(hicpp-signed-bitwise)
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = old | arg; // NOLINT(hicpp-signed-bitwise)
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = old ^ arg; // NOLINT(hicpp-signed-bitwise)
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = std::min(old, arg);
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address,
                 const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = std::max(old, arg);
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address,
                     const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = (old >= arg) ? T(0) : old + T(1);
    }

    // Implicit flush due to omp critical

    return old;
}


template <typename T, typename>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address,
                     const T arg)
{
    // Implicit flush due to omp critical

    T old;
    #pragma omp critical
    {
        old = *address;
        *address = (old == T(0) || old > arg) ? arg : old - T(1);
    }

    // Implicit flush due to omp critical

    return old;
}

} // namespace openmp

} // namespace stdgpu



#endif // STDGPU_OPENMP_ATOMIC_DETAIL_H
