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

#include <stdgpu/limits.h>

namespace stdgpu
{

#define STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(T)                                                                       \
    constexpr bool numeric_limits<T>::is_specialized;                                                                  \
    constexpr bool numeric_limits<T>::is_signed;                                                                       \
    constexpr bool numeric_limits<T>::is_integer;                                                                      \
    constexpr bool numeric_limits<T>::is_exact;                                                                        \
    constexpr int numeric_limits<T>::digits;                                                                           \
    constexpr int numeric_limits<T>::radix;

STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(bool)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(char)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(signed char)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(unsigned char)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(wchar_t)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(char16_t)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(char32_t)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(short)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(unsigned short)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(int)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(unsigned int)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(long)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(unsigned long)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(long long)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(unsigned long long)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(float)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(double)
STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS(long double)

#undef STDGPU_DETAIL_COMPOUND_NUMERIC_LIMITS

} // namespace stdgpu
