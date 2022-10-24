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

#ifndef STDGPU_LIMITS_DETAIL_H
#define STDGPU_LIMITS_DETAIL_H

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>

#include <stdgpu/compiler.h>

namespace stdgpu
{

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::min() noexcept
{
    return T();
}

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::max() noexcept
{
    return T();
}

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::lowest() noexcept
{
    return T();
}

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::epsilon() noexcept
{
    return T();
}

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::round_error() noexcept
{
    return T();
}

template <typename T>
constexpr STDGPU_HOST_DEVICE T
numeric_limits<T>::infinity() noexcept
{
    return T();
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::min() noexcept
{
    return false;
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::max() noexcept
{
    return true;
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::lowest() noexcept
{
    return false;
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::epsilon() noexcept
{
    return false;
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::round_error() noexcept
{
    return false;
}

constexpr STDGPU_HOST_DEVICE bool
numeric_limits<bool>::infinity() noexcept
{
    return false;
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::min() noexcept
{
    return CHAR_MIN;
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::max() noexcept
{
    return CHAR_MAX;
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char
numeric_limits<char>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::min() noexcept
{
    return SCHAR_MIN;
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::max() noexcept
{
    return SCHAR_MAX;
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE signed char
numeric_limits<signed char>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::max() noexcept
{
    return UCHAR_MAX;
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned char
numeric_limits<unsigned char>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::min() noexcept
{
    return WCHAR_MIN;
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::max() noexcept
{
    return WCHAR_MAX;
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE wchar_t
numeric_limits<wchar_t>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::max() noexcept
{
    return UINT_LEAST16_MAX;
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char16_t
numeric_limits<char16_t>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::max() noexcept
{
    return UINT_LEAST32_MAX;
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE char32_t
numeric_limits<char32_t>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::min() noexcept
{
    return SHRT_MIN;
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::max() noexcept
{
    return SHRT_MAX;
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE short
numeric_limits<short>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::max() noexcept
{
    return USHRT_MAX;
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned short
numeric_limits<unsigned short>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::min() noexcept
{
    return INT_MIN;
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::max() noexcept
{
    return INT_MAX;
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE int
numeric_limits<int>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::max() noexcept
{
    return UINT_MAX;
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned int
numeric_limits<unsigned int>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::min() noexcept
{
    return LONG_MIN;
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::max() noexcept
{
    return LONG_MAX;
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long
numeric_limits<long>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::max() noexcept
{
    return ULONG_MAX;
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long
numeric_limits<unsigned long>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::min() noexcept
{
    return LLONG_MIN;
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::max() noexcept
{
    return LLONG_MAX;
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE long long
numeric_limits<long long>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::min() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::max() noexcept
{
    return ULLONG_MAX;
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::lowest() noexcept
{
    return min();
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::epsilon() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::round_error() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE unsigned long long
numeric_limits<unsigned long long>::infinity() noexcept
{
    return 0;
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::min() noexcept
{
    return FLT_MIN;
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::max() noexcept
{
    return FLT_MAX;
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::lowest() noexcept
{
    return -FLT_MAX;
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::epsilon() noexcept
{
    return FLT_EPSILON;
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::round_error() noexcept
{
    return 0.5F; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

constexpr STDGPU_HOST_DEVICE float
numeric_limits<float>::infinity() noexcept
{
    return HUGE_VALF;
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::min() noexcept
{
    return DBL_MIN;
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::max() noexcept
{
    return DBL_MAX;
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::lowest() noexcept
{
    return -DBL_MAX;
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::epsilon() noexcept
{
    return DBL_EPSILON;
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::round_error() noexcept
{
    return 0.5; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

constexpr STDGPU_HOST_DEVICE double
numeric_limits<double>::infinity() noexcept
{
    return HUGE_VAL;
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::min() noexcept
{
    return LDBL_MIN;
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::max() noexcept
{
    return LDBL_MAX;
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::lowest() noexcept
{
    return -LDBL_MAX;
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::epsilon() noexcept
{
    return LDBL_EPSILON;
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::round_error() noexcept
{
    return 0.5L; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

constexpr STDGPU_HOST_DEVICE long double
numeric_limits<long double>::infinity() noexcept
{
// Suppress long double is treated as double in device code warning for MSVC on CUDA
#if STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
    return HUGE_VAL;
#else
    return HUGE_VALL;
#endif
}

} // namespace stdgpu

#endif // STDGPU_LIMITS_DETAIL_H
