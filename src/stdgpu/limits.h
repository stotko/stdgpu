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

#ifndef STDGPU_LIMITS_H
#define STDGPU_LIMITS_H

/**
 * \addtogroup limits limits
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/limits.h
 */

#include <cfloat>
#include <climits>

#include <stdgpu/compiler.h>
#include <stdgpu/platform.h>

namespace stdgpu
{

/**
 * \ingroup limits
 * \brief Generic traits
 * \tparam T The type for which limits should be specified
 */
template <class T>
struct numeric_limits
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE T
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE T
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE T
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE T
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE T
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE T
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = false;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = false;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = false;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = 0;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 0;
};

/**
 * \ingroup limits
 * \brief Specialization for bool
 */
template <>
struct numeric_limits<bool>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE bool
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE bool
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE bool
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE bool
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE bool
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE bool
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for char
 */
template <>
struct numeric_limits<char>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE char
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE char
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE char
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     * \note implementation-defined
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT - static_cast<int>(numeric_limits<char>::is_signed);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for signed char
 */
template <>
struct numeric_limits<signed char>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE signed char
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT - 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for unsigned char
 */
template <>
struct numeric_limits<unsigned char>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned char
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for wchar_t
 */
template <>
struct numeric_limits<wchar_t>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE wchar_t
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

/**
 * \var is_signed
 * \brief Whether the type is signed
 * \note implementation-defined
 */
#if STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
    static constexpr bool is_signed = false;
#else
    static constexpr bool is_signed = true;
#endif

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(wchar_t) - static_cast<int>(numeric_limits<wchar_t>::is_signed);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for char16_t
 */
template <>
struct numeric_limits<char16_t>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE char16_t
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(char16_t);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for char32_t
 */
template <>
struct numeric_limits<char32_t>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE char32_t
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(char32_t);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for short
 */
template <>
struct numeric_limits<short>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE short
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE short
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE short
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE short
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE short
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE short
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(short) - 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for unsigned short
 */
template <>
struct numeric_limits<unsigned short>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned short
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(unsigned short);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for int
 */
template <>
struct numeric_limits<int>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE int
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE int
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE int
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE int
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE int
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE int
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(int) - 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for unsigned int
 */
template <>
struct numeric_limits<unsigned int>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned int
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(unsigned int);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for long
 */
template <>
struct numeric_limits<long>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE long
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE long
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE long
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(long) - 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for unsigned long
 */
template <>
struct numeric_limits<unsigned long>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(unsigned long);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for long long
 */
template <>
struct numeric_limits<long long>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long long
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long long
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long long
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE long long
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE long long
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE long long
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(long long) - 1;

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for unsigned long long
 */
template <>
struct numeric_limits<unsigned long long>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE unsigned long long
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = false;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = true;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = true;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = CHAR_BIT * sizeof(unsigned long long);

    /**
     * \brief Integer base
     */
    static constexpr int radix = 2;
};

/**
 * \ingroup limits
 * \brief Specialization for float
 */
template <>
struct numeric_limits<float>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE float
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE float
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE float
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE float
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE float
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE float
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = false;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = false;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = FLT_MANT_DIG;

    /**
     * \brief Integer base
     */
    static constexpr int radix = FLT_RADIX;
};

/**
 * \ingroup limits
 * \brief Specialization for double
 */
template <>
struct numeric_limits<double>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE double
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE double
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE double
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE double
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE double
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE double
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = false;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = false;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = DBL_MANT_DIG;

    /**
     * \brief Integer base
     */
    static constexpr int radix = FLT_RADIX;
};

/**
 * \ingroup limits
 * \brief Specialization for long double
 */
template <>
struct numeric_limits<long double>
{
    /**
     * \brief Smallest representable finite value
     * \return Smallest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long double
    min() noexcept;

    /**
     * \brief Largest representable finite value
     * \return Largest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long double
    max() noexcept;

    /**
     * \brief Lowest representable finite value
     * \return Lowest representable finite value
     */
    static constexpr STDGPU_HOST_DEVICE long double
    lowest() noexcept;

    /**
     * \brief Machine epsilon
     * \return Machine epsilon
     */
    static constexpr STDGPU_HOST_DEVICE long double
    epsilon() noexcept;

    /**
     * \brief Maximum round error
     * \return Maximum round error
     */
    static constexpr STDGPU_HOST_DEVICE long double
    round_error() noexcept;

    /**
     * \brief Infinity value
     * \return Infinity value
     */
    static constexpr STDGPU_HOST_DEVICE long double
    infinity() noexcept;

    /**
     * \brief Whether the traits of the type are specialized
     */
    static constexpr bool is_specialized = true;

    /**
     * \brief Whether the type is signed
     */
    static constexpr bool is_signed = true;

    /**
     * \brief Whether the type is an integer
     */
    static constexpr bool is_integer = false;

    /**
     * \brief Whether the type is exact
     */
    static constexpr bool is_exact = false;

    /**
     * \brief Number of radix digits
     */
    static constexpr int digits = LDBL_MANT_DIG;

    /**
     * \brief Integer base
     */
    static constexpr int radix = FLT_RADIX;
};

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/limits_detail.h>

#endif // STDGPU_LIMITS_H
