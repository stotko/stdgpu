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

#ifndef STDGPU_BIT_DETAIL_H
#define STDGPU_BIT_DETAIL_H

#if STDGPU_BACKEND == STDGPU_BACKEND_CUDA && STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
    #define STDGPU_BACKEND_BIT_HEADER <stdgpu/STDGPU_BACKEND_DIRECTORY/bit.cuh>
    #include STDGPU_BACKEND_BIT_HEADER
    #undef STDGPU_BACKEND_BIT_HEADER
#endif
#include <stdgpu/contract.h>
#include <stdgpu/limits.h>



namespace stdgpu
{

namespace detail
{

template <typename T>
STDGPU_HOST_DEVICE T
log2pow2(T number)
{
    T result = 0;
    T shifted_number = number;
    while (shifted_number >>= 1)
    {
        ++result;
    }
    return result;
}

template <typename T>
STDGPU_HOST_DEVICE int
popcount(T number)
{
    int result;
    for (result = 0; number; ++result)
    {
        // Clear the least significant bit set
        number &= number - 1;
    }
    return result;
}

} // namespace detail

template <typename T, typename>
STDGPU_HOST_DEVICE bool
ispow2(const T number)
{
    return ((number != 0) && !(number & (number - 1)));
}


template <typename T, typename>
STDGPU_HOST_DEVICE T
ceil2(const T number)
{
    T result = number;

    // Special case zero
    result += (result == 0);

    result--;
    for (index_t i = 0; i < stdgpu::numeric_limits<T>::digits; ++i)
    {
        result |= result >> i;
    }
    result++;

    // If result is not representable in T, we have undefined behavior
    // --> In this case, we have an overflow to 0
    STDGPU_ENSURES(result == 0 || ispow2(result));

    return result;
}


template <typename T, typename>
STDGPU_HOST_DEVICE T
floor2(const T number)
{
    // Special case zero
    if (number == 0) return 0;

    T result = number;
    for (index_t i = 0; i < stdgpu::numeric_limits<T>::digits; ++i)
    {
        result |= result >> i;
    }
    result &= ~(result >> 1);

    STDGPU_ENSURES(ispow2(result));

    return result;
}


template <typename T, typename>
STDGPU_HOST_DEVICE T
mod2(const T number,
     const T divider)
{
    STDGPU_EXPECTS(ispow2(divider));

    T result = number & (divider - 1);

    STDGPU_ENSURES(result < divider);

    return result;
}


template <>
inline STDGPU_HOST_DEVICE unsigned int
log2pow2<unsigned int>(const unsigned int number)
{
    STDGPU_EXPECTS(ispow2(number));

    #if STDGPU_BACKEND == STDGPU_BACKEND_CUDA && STDGPU_CODE == STDGPU_CODE_DEVICE
        return stdgpu::STDGPU_BACKEND_NAMESPACE::log2pow2(number);
    #else
        return detail::log2pow2(number);
    #endif
}

template <>
inline STDGPU_HOST_DEVICE unsigned long long int
log2pow2<unsigned long long int>(const unsigned long long int number)
{
    STDGPU_EXPECTS(ispow2(number));

    #if STDGPU_BACKEND == STDGPU_BACKEND_CUDA && STDGPU_CODE == STDGPU_CODE_DEVICE
        return stdgpu::STDGPU_BACKEND_NAMESPACE::log2pow2(number);
    #else
        return detail::log2pow2(number);
    #endif
}

template <typename T, typename>
STDGPU_HOST_DEVICE T
log2pow2(const T number)
{
    STDGPU_EXPECTS(ispow2(number));

    return detail::log2pow2(number);
}


template <>
inline STDGPU_HOST_DEVICE int
popcount<unsigned int>(const unsigned int number)
{
    #if STDGPU_BACKEND == STDGPU_BACKEND_CUDA && STDGPU_CODE == STDGPU_CODE_DEVICE
        return stdgpu::STDGPU_BACKEND_NAMESPACE::popcount(number);
    #else
        return detail::popcount(number);
    #endif
}


template <>
inline STDGPU_HOST_DEVICE int
popcount<unsigned long long int>(const unsigned long long int number)
{
    #if STDGPU_BACKEND == STDGPU_BACKEND_CUDA && STDGPU_CODE == STDGPU_CODE_DEVICE
        return stdgpu::STDGPU_BACKEND_NAMESPACE::popcount(number);
    #else
        return detail::popcount(number);
    #endif
}


template <typename T, typename>
STDGPU_HOST_DEVICE int
popcount(const T number)
{
    return detail::popcount(number);
}

} // namespace stdgpu



#endif // STDGPU_BIT_DETAIL_H
