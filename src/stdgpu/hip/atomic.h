/*
 *  Copyright 2020 Patrick Stotko
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

#ifndef STDGPU_HIP_ATOMIC_H
#define STDGPU_HIP_ATOMIC_H


#include <type_traits>



namespace stdgpu
{

namespace hip
{

/**
 * \brief Atomically loads and returns the current value of the atomic object
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \return The current value of this object
 */
template <typename T>
STDGPU_DEVICE_ONLY T
atomic_load(T* address);

/**
 * \brief Atomically replaces the current value with desired one
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] desired The value to store to the atomic object
 */
template <typename T>
STDGPU_DEVICE_ONLY void
atomic_store(T* address,
             const T desired);

/**
 * \brief Atomically exchanges the stored value with the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] desired The desired argument to store
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_exchange(T* address,
                const T desired);

/**
 * \brief Atomically exchanges the stored value with the given argument if it equals the expected value
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] expected The expected stored value
 * \param[in] desired The desired argument to store
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_compare_exchange(T* address,
                        const T expected,
                        const T desired);

/**
 * \brief Atomically computes and stores the addition of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of addition
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the subtraction of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of subtraction
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of bitwise AND
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of bitwise OR
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of bitwise XOR
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the minimum of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of minimum
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_min(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the maximum of the stored value and the given argument
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of maximum
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_max(T* address,
                 const T arg);

/**
 * \brief Atomically computes and stores the incrementation of the value and modulus with arg
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of modulus
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_same<T, unsigned int>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_inc_mod(T* address,
                     const T arg);

/**
 * \brief Atomically computes and stores the decrementation of the value and modulus with arg
 * \tparam T The type of the atomically managed object
 * \param[in] address A pointer to a value
 * \param[in] arg The other argument of modulus
 * \return The old value
 */
template <typename T, typename = std::enable_if_t<std::is_same<T, unsigned int>::value>>
STDGPU_DEVICE_ONLY T
atomic_fetch_dec_mod(T* address,
                     const T arg);

} // namespace hip

} // namespace stdgpu



#include <stdgpu/hip/impl/atomic_detail.h>



#endif // STDGPU_HIP_ATOMIC_H
