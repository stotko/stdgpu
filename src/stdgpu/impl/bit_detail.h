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

#include <stdgpu/contract.h>
#include <stdgpu/limits.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

template <typename T, typename>
STDGPU_HOST_DEVICE bool
ispow2(const T number)
{
    return ((number != 0) && !(number & (number - 1)));
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


template <typename T, typename>
STDGPU_HOST_DEVICE T
log2pow2(const T number)
{
    STDGPU_EXPECTS(ispow2(number));

    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        return __ffsll(number) - 1;
    #else
        // Similar intrinsics for the host part are avoided to ensure cross compilation
        T result = 0;
        T shifted_number = number;
        while (shifted_number >>= 1)
        {
            ++result;
        }
        return result;
    #endif
}


namespace detail
{
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
}


template <>
inline STDGPU_HOST_DEVICE int
popcount<unsigned int>(const unsigned int number)
{
    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        return __popc(number);
    #else
        // Similar intrinsics for the host part are avoided to ensure cross compilation
        return detail::popcount(number);
    #endif
}


template <>
inline STDGPU_HOST_DEVICE int
popcount<unsigned long long int>(const unsigned long long int number)
{
    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        return __popcll(number);
    #else
        // Similar intrinsics for the host part are avoided to ensure cross compilation
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
