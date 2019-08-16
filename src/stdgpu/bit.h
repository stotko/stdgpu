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

#ifndef STDGPU_BIT_H
#define STDGPU_BIT_H

/**
 * \file stdgpu/bit.h
 */

#include <type_traits>

#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \brief Determines whether the number is a power of two
 * \param[in] number A number
 * \return True if number is a power of two, false otherwise
 */
template <typename T, typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
STDGPU_HOST_DEVICE bool
ispow2(const T number);

/**
 * \brief Computes the modulus of the given number and a power of two divider
 * \param[in] number A number
 * \param[in] divider The divider with divider = 2^n
 * \return The modulos of the given number and divider
 * \pre ispow2(divider)
 * \post result >= 0
 * \post result < divider
 */
template <typename T, typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
STDGPU_HOST_DEVICE T
mod2(const T number,
     const T divider);

/**
 * \brief Computes the base-2 logarithm of a power of two
 * \param[in] number A number
 * \return The base-2 logarithm of the number
 * \pre ispow2(divider)
 */
template <typename T, typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
STDGPU_HOST_DEVICE T
log2pow2(const T number);

/**
 * \brief Counts the number of set bits in the number
 * \param[in] number A number
 * \return The number of set bits
 */
template <typename T, typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
STDGPU_HOST_DEVICE int
popcount(const T number);

} // namespace stdgpu



#include <stdgpu/impl/bit_detail.h>



#endif // STDGPU_BIT_H
