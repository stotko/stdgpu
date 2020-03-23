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
#include <stdgpu/platform.h>



namespace stdgpu
{

template <class T>
struct numeric_limits
{
    static_assert(sizeof(T) != sizeof(T), "stdgpu::numeric_limits : No specialization for type T provided");
};


template <>
struct numeric_limits<bool>
{
    static constexpr STDGPU_HOST_DEVICE bool min() noexcept                         { return false; }
    static constexpr STDGPU_HOST_DEVICE bool max() noexcept                         { return true; }
    static constexpr STDGPU_HOST_DEVICE bool lowest() noexcept                      { return false; }
    static constexpr STDGPU_HOST_DEVICE bool epsilon() noexcept                     { return false; }
    static constexpr STDGPU_HOST_DEVICE bool round_error() noexcept                 { return false; }
    static constexpr STDGPU_HOST_DEVICE bool infinity() noexcept                    { return false; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<char>
{
    static constexpr STDGPU_HOST_DEVICE char min() noexcept                         { return CHAR_MIN; }
    static constexpr STDGPU_HOST_DEVICE char max() noexcept                         { return CHAR_MAX; }
    static constexpr STDGPU_HOST_DEVICE char lowest() noexcept                      { return min(); }
    static constexpr STDGPU_HOST_DEVICE char epsilon() noexcept                     { return 0; }
    static constexpr STDGPU_HOST_DEVICE char round_error() noexcept                 { return 0; }
    static constexpr STDGPU_HOST_DEVICE char infinity() noexcept                    { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT - numeric_limits<char>::is_signed;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<signed char>
{
    static constexpr STDGPU_HOST_DEVICE signed char min() noexcept                  { return SCHAR_MIN; }
    static constexpr STDGPU_HOST_DEVICE signed char max() noexcept                  { return SCHAR_MAX; }
    static constexpr STDGPU_HOST_DEVICE signed char lowest() noexcept               { return min(); }
    static constexpr STDGPU_HOST_DEVICE signed char epsilon() noexcept              { return 0; }
    static constexpr STDGPU_HOST_DEVICE signed char round_error() noexcept          { return 0; }
    static constexpr STDGPU_HOST_DEVICE signed char infinity() noexcept             { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT - 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<unsigned char>
{
    static constexpr STDGPU_HOST_DEVICE unsigned char min() noexcept                { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned char max() noexcept                { return UCHAR_MAX; }
    static constexpr STDGPU_HOST_DEVICE unsigned char lowest() noexcept             { return min(); }
    static constexpr STDGPU_HOST_DEVICE unsigned char epsilon() noexcept            { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned char round_error() noexcept        { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned char infinity() noexcept           { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<wchar_t>
{
    static constexpr STDGPU_HOST_DEVICE wchar_t min() noexcept                      { return WCHAR_MIN; }
    static constexpr STDGPU_HOST_DEVICE wchar_t max() noexcept                      { return WCHAR_MAX; }
    static constexpr STDGPU_HOST_DEVICE wchar_t lowest() noexcept                   { return min(); }
    static constexpr STDGPU_HOST_DEVICE wchar_t epsilon() noexcept                  { return 0; }
    static constexpr STDGPU_HOST_DEVICE wchar_t round_error() noexcept              { return 0; }
    static constexpr STDGPU_HOST_DEVICE wchar_t infinity() noexcept                 { return 0; }
    static constexpr bool is_specialized                                            = true;
    #if STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
        static constexpr bool is_signed                                             = false;
    #else
        static constexpr bool is_signed                                             = true;
    #endif
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(wchar_t) - numeric_limits<wchar_t>::is_signed;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<char16_t>
{
    static constexpr STDGPU_HOST_DEVICE char16_t min() noexcept                     { return 0; }
    static constexpr STDGPU_HOST_DEVICE char16_t max() noexcept                     { return UINT_LEAST16_MAX; }
    static constexpr STDGPU_HOST_DEVICE char16_t lowest() noexcept                  { return min(); }
    static constexpr STDGPU_HOST_DEVICE char16_t epsilon() noexcept                 { return 0; }
    static constexpr STDGPU_HOST_DEVICE char16_t round_error() noexcept             { return 0; }
    static constexpr STDGPU_HOST_DEVICE char16_t infinity() noexcept                { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(char16_t);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<char32_t>
{
    static constexpr STDGPU_HOST_DEVICE char32_t min() noexcept                     { return 0; }
    static constexpr STDGPU_HOST_DEVICE char32_t max() noexcept                     { return UINT_LEAST32_MAX; }
    static constexpr STDGPU_HOST_DEVICE char32_t lowest() noexcept                  { return min(); }
    static constexpr STDGPU_HOST_DEVICE char32_t epsilon() noexcept                 { return 0; }
    static constexpr STDGPU_HOST_DEVICE char32_t round_error() noexcept             { return 0; }
    static constexpr STDGPU_HOST_DEVICE char32_t infinity() noexcept                { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(char32_t);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<short>
{
    static constexpr STDGPU_HOST_DEVICE short min() noexcept                        { return SHRT_MIN; }
    static constexpr STDGPU_HOST_DEVICE short max() noexcept                        { return SHRT_MAX; }
    static constexpr STDGPU_HOST_DEVICE short lowest() noexcept                     { return min(); }
    static constexpr STDGPU_HOST_DEVICE short epsilon() noexcept                    { return 0; }
    static constexpr STDGPU_HOST_DEVICE short round_error() noexcept                { return 0; }
    static constexpr STDGPU_HOST_DEVICE short infinity() noexcept                   { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(short) - 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<unsigned short>
{
    static constexpr STDGPU_HOST_DEVICE unsigned short min() noexcept               { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned short max() noexcept               { return USHRT_MAX; }
    static constexpr STDGPU_HOST_DEVICE unsigned short lowest() noexcept            { return min(); }
    static constexpr STDGPU_HOST_DEVICE unsigned short epsilon() noexcept           { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned short round_error() noexcept       { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned short infinity() noexcept          { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(short);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<int>
{
    static constexpr STDGPU_HOST_DEVICE int min() noexcept                          { return INT_MIN; }
    static constexpr STDGPU_HOST_DEVICE int max() noexcept                          { return INT_MAX; }
    static constexpr STDGPU_HOST_DEVICE int lowest() noexcept                       { return min(); }
    static constexpr STDGPU_HOST_DEVICE int epsilon() noexcept                      { return 0; }
    static constexpr STDGPU_HOST_DEVICE int round_error() noexcept                  { return 0; }
    static constexpr STDGPU_HOST_DEVICE int infinity() noexcept                     { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(int) - 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<unsigned int>
{
    static constexpr STDGPU_HOST_DEVICE unsigned int min() noexcept                 { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned int max() noexcept                 { return UINT_MAX; }
    static constexpr STDGPU_HOST_DEVICE unsigned int lowest() noexcept              { return min(); }
    static constexpr STDGPU_HOST_DEVICE unsigned int epsilon() noexcept             { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned int round_error() noexcept         { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned int infinity() noexcept            { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(int);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<long>
{
    static constexpr STDGPU_HOST_DEVICE long min() noexcept                         { return LONG_MIN; }
    static constexpr STDGPU_HOST_DEVICE long max() noexcept                         { return LONG_MAX; }
    static constexpr STDGPU_HOST_DEVICE long lowest() noexcept                      { return min(); }
    static constexpr STDGPU_HOST_DEVICE long epsilon() noexcept                     { return 0; }
    static constexpr STDGPU_HOST_DEVICE long round_error() noexcept                 { return 0; }
    static constexpr STDGPU_HOST_DEVICE long infinity() noexcept                    { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(long) - 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<unsigned long>
{
    static constexpr STDGPU_HOST_DEVICE unsigned long min() noexcept                { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long max() noexcept                { return ULONG_MAX; }
    static constexpr STDGPU_HOST_DEVICE unsigned long lowest() noexcept             { return min(); }
    static constexpr STDGPU_HOST_DEVICE unsigned long epsilon() noexcept            { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long round_error() noexcept        { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long infinity() noexcept           { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(long);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<long long>
{
    static constexpr STDGPU_HOST_DEVICE long long min() noexcept                    { return LLONG_MIN; }
    static constexpr STDGPU_HOST_DEVICE long long max() noexcept                    { return LLONG_MAX; }
    static constexpr STDGPU_HOST_DEVICE long long lowest() noexcept                 { return min(); }
    static constexpr STDGPU_HOST_DEVICE long long epsilon() noexcept                { return 0; }
    static constexpr STDGPU_HOST_DEVICE long long round_error() noexcept            { return 0; }
    static constexpr STDGPU_HOST_DEVICE long long infinity() noexcept               { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(long long) - 1;
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<unsigned long long>
{
    static constexpr STDGPU_HOST_DEVICE unsigned long long min() noexcept           { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long long max() noexcept           { return ULLONG_MAX; }
    static constexpr STDGPU_HOST_DEVICE unsigned long long lowest() noexcept        { return min(); }
    static constexpr STDGPU_HOST_DEVICE unsigned long long epsilon() noexcept       { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long long round_error() noexcept   { return 0; }
    static constexpr STDGPU_HOST_DEVICE unsigned long long infinity() noexcept      { return 0; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = false;
    static constexpr bool is_integer                                                = true;
    static constexpr bool is_exact                                                  = true;
    static constexpr int digits                                                     = CHAR_BIT * sizeof(long long);
    static constexpr int radix                                                      = 2;
};


template <>
struct numeric_limits<float>
{
    static constexpr STDGPU_HOST_DEVICE float min() noexcept                        { return FLT_MIN; }
    static constexpr STDGPU_HOST_DEVICE float max() noexcept                        { return FLT_MAX; }
    static constexpr STDGPU_HOST_DEVICE float lowest() noexcept                     { return -FLT_MAX; }
    static constexpr STDGPU_HOST_DEVICE float epsilon() noexcept                    { return FLT_EPSILON; }
    static constexpr STDGPU_HOST_DEVICE float round_error() noexcept                { return 0.5F; }
    static constexpr STDGPU_HOST_DEVICE float infinity() noexcept                   { return HUGE_VALF; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = false;
    static constexpr bool is_exact                                                  = false;
    static constexpr int digits                                                     = FLT_MANT_DIG;
    static constexpr int radix                                                      = FLT_RADIX;
};


template <>
struct numeric_limits<double>
{
    static constexpr STDGPU_HOST_DEVICE double min() noexcept                       { return DBL_MIN; }
    static constexpr STDGPU_HOST_DEVICE double max() noexcept                       { return DBL_MAX; }
    static constexpr STDGPU_HOST_DEVICE double lowest() noexcept                    { return -DBL_MAX; }
    static constexpr STDGPU_HOST_DEVICE double epsilon() noexcept                   { return DBL_EPSILON; }
    static constexpr STDGPU_HOST_DEVICE double round_error() noexcept               { return 0.5; }
    static constexpr STDGPU_HOST_DEVICE double infinity() noexcept                  { return HUGE_VAL; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = false;
    static constexpr bool is_exact                                                  = false;
    static constexpr int digits                                                     = DBL_MANT_DIG;
    static constexpr int radix                                                      = FLT_RADIX;
};


template <>
struct numeric_limits<long double>
{
    static constexpr STDGPU_HOST_DEVICE long double min() noexcept                  { return LDBL_MIN; }
    static constexpr STDGPU_HOST_DEVICE long double max() noexcept                  { return LDBL_MAX; }
    static constexpr STDGPU_HOST_DEVICE long double lowest() noexcept               { return -LDBL_MAX; }
    static constexpr STDGPU_HOST_DEVICE long double epsilon() noexcept              { return LDBL_EPSILON; }
    static constexpr STDGPU_HOST_DEVICE long double round_error() noexcept          { return 0.5L; }
    static constexpr STDGPU_HOST_DEVICE long double infinity() noexcept             { return HUGE_VALL; }
    static constexpr bool is_specialized                                            = true;
    static constexpr bool is_signed                                                 = true;
    static constexpr bool is_integer                                                = false;
    static constexpr bool is_exact                                                  = false;
    static constexpr int digits                                                     = LDBL_MANT_DIG;
    static constexpr int radix                                                      = FLT_RADIX;
};

} // namespace stdgpu



#endif // STDGPU_LIMITS_DETAIL_H
