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

#include <cstddef>

#include <stdgpu/contract.h>
#include <stdgpu/limits.h>

namespace stdgpu
{

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE bool
has_single_bit(const T number) noexcept
{
    return ((number != 0) && !(number & (number - static_cast<T>(1))));
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE T
bit_ceil(const T number) noexcept
{
    T result = number;

    // Special case zero
    result += (result == 0);

    result--;
    for (index_t i = 0; i < numeric_limits<T>::digits; ++i)
    {
        result |= result >> static_cast<T>(i);
    }
    result++;

    // If result is not representable in T, we have undefined behavior
    // --> In this case, we have an overflow to 0
    STDGPU_ENSURES(result == 0 || has_single_bit(result));

    return result;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE T
bit_floor(const T number) noexcept
{
    // Special case zero
    if (number == 0)
    {
        return 0;
    }

    T result = number;
    for (index_t i = 0; i < numeric_limits<T>::digits; ++i)
    {
        result |= result >> static_cast<T>(i);
    }
    result &= ~(result >> static_cast<T>(1));

    STDGPU_ENSURES(has_single_bit(result));

    return result;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE T
bit_mod(const T number, const T divider) noexcept
{
    STDGPU_EXPECTS(has_single_bit(divider));

    T result = number & (divider - 1);

    STDGPU_ENSURES(result < divider);

    return result;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE T
bit_width(const T number) noexcept
{
    if (number == 0)
    {
        return 0;
    }

    T result = 1;
    T shifted_number = number;
    while (shifted_number >>= static_cast<T>(1))
    {
        ++result;
    }

    STDGPU_ENSURES(result <= static_cast<T>(numeric_limits<T>::digits));

    return result;
}

template <typename T, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_unsigned_v<T>)>
STDGPU_HOST_DEVICE int
popcount(const T number) noexcept
{
    int result = 0;
    T cleared_number = number;
    for (; cleared_number; ++result)
    {
        // Clear the least significant bit set
        cleared_number &= cleared_number - static_cast<T>(1);
    }

    STDGPU_ENSURES(0 <= result);
    STDGPU_ENSURES(result <= numeric_limits<T>::digits);

    return result;
}

template <typename To,
          typename From,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<To> &&
                                               std::is_trivially_copyable_v<From>)>
STDGPU_HOST_DEVICE To
bit_cast(const From& object) noexcept
{
    To result;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* result_bytes = reinterpret_cast<std::byte*>(&result);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* object_bytes = reinterpret_cast<const std::byte*>(&object);

    for (std::size_t i = 0; i < sizeof(To); ++i)
    {
        result_bytes[i] = object_bytes[i];
    }

    return result;
}

} // namespace stdgpu

#endif // STDGPU_BIT_DETAIL_H
