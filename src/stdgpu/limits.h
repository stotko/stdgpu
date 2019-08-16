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
 * \file stdgpu/limits.h
 */



namespace stdgpu
{

/**
 * \brief Generic declaration
 * \tparam T The type for which limits should be specified
 */
template <class T>
struct numeric_limits;


/**
 * \brief Specialization for bool
 */
template <>
struct numeric_limits<bool>;

/**
 * \brief Specialization for char
 */
template <>
struct numeric_limits<char>;

/**
 * \brief Specialization for signed char
 */
template <>
struct numeric_limits<signed char>;

/**
 * \brief Specialization for unsigned char
 */
template <>
struct numeric_limits<unsigned char>;

/**
 * \brief Specialization for wchar_t
 */
template <>
struct numeric_limits<wchar_t>;

/**
 * \brief Specialization for char16_t
 */
template <>
struct numeric_limits<char16_t>;

/**
 * \brief Specialization for char32_t
 */
template <>
struct numeric_limits<char32_t>;

/**
 * \brief Specialization for short
 */
template <>
struct numeric_limits<short>;

/**
 * \brief Specialization for unsigned short
 */
template <>
struct numeric_limits<unsigned short>;

/**
 * \brief Specialization for int
 */
template <>
struct numeric_limits<int>;

/**
 * \brief Specialization for unsigned int
 */
template <>
struct numeric_limits<unsigned int>;

/**
 * \brief Specialization for long
 */
template <>
struct numeric_limits<long>;

/**
 * \brief Specialization for unsigned long
 */
template <>
struct numeric_limits<unsigned long>;

/**
 * \brief Specialization for long long
 */
template <>
struct numeric_limits<long long>;

/**
 * \brief Specialization for unsigned long long
 */
template <>
struct numeric_limits<unsigned long long>;

/**
 * \brief Specialization for float
 */
template <>
struct numeric_limits<float>;

/**
 * \brief Specialization for double
 */
template <>
struct numeric_limits<double>;

/**
 * \brief Specialization for long double
 */
template <>
struct numeric_limits<long double>;

} // namespace stdgpu



#include <stdgpu/impl/limits_detail.h>



#endif // STDGPU_LIMITS_H
