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

#ifndef STDGPU_ATOMIC_H
#define STDGPU_ATOMIC_H

#include <stdgpu/impl/platform_check.h>

/**
 * \addtogroup atomic atomic
 * \ingroup data_structures
 * @{
 */

/**
 * \file stdgpu/atomic.cuh
 */

#include <cstddef>
#include <type_traits>

#include <stdgpu/impl/type_traits.h>
#include <stdgpu/platform.h>

///////////////////////////////////////////////////////////

#include <stdgpu/atomic_fwd>

///////////////////////////////////////////////////////////

namespace stdgpu
{

/**
 * \ingroup atomic
 * \brief The memory order types for atomic operations
 */
enum memory_order
{
    memory_order_relaxed, /**< memory_order_relaxed */
    memory_order_consume, /**< memory_order_consume */
    memory_order_acquire, /**< memory_order_acquire */
    memory_order_release, /**< memory_order_release */
    memory_order_acq_rel, /**< memory_order_acq_rel */
    memory_order_seq_cst  /**< memory_order_seq_cst */
};

/**
 * \ingroup atomic
 * \brief A synchronization fence enforcing the given memory order
 * \param[in] order The memory order
 * \note The synchronization might be stricter than requested
 */
STDGPU_DEVICE_ONLY void
atomic_thread_fence(const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief A synchronization fence enforcing the given memory order. Similar to atomic_thread_fence
 * \param[in] order The memory order
 * \note The synchronization might be stricter than requested
 */
STDGPU_DEVICE_ONLY void
atomic_signal_fence(const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief A class to model an atomic object of type T on the GPU
 * \tparam T The type of the atomically managed object
 * \tparam Allocator The allocator type
 *
 * Supported types:
 *  - unsigned int
 *  - int
 *  - unsigned long long int
 *  - float (experimental)
 *
 * Differences to std::atomic:
 *  - Atomics must be modeled as containers since threads have to operate on the exact same object (which also requires
 * copy and move constructors)
 *  - Manual allocation and destruction of container required
 *  - All operations (including load() and store()) may follow stricter ordering than requested
 *  - Additional min and max functions for all supported integer and floating point types
 *  - Additional increment/decrement + modulo functions for unsigned int
 */
template <typename T, typename Allocator>
class atomic
{
public:
    static_assert(std::is_same_v<T, unsigned int> || std::is_same_v<T, int> ||
                          std::is_same_v<T, unsigned long long int> || std::is_same_v<T, float>,
                  "stdgpu::atomic: No support for type T");

    using value_type = T;               /**< T */
    using difference_type = value_type; /**< value_type */

    using allocator_type = Allocator; /**< Allocator */

    /**
     * \brief Creates an object of this class on the GPU (device)
     * \param[in] allocator The allocator instance to use
     * \return A newly created object of this class allocated on the GPU (device)
     * \note The size is implicitly set to 1 (and not needed as a parameter) as the object only manages a single value
     */
    static atomic
    createDeviceObject(const Allocator& allocator = Allocator());

    /**
     * \brief Destroys the given object of this class on the GPU (device)
     * \param[in] device_object The object allocated on the GPU (device)
     */
    static void
    destroyDeviceObject(atomic& device_object);

    /**
     * \brief Empty constructor
     */
    atomic() noexcept;

    /**
     * \brief Returns the container allocator
     * \return The container allocator
     */
    STDGPU_HOST_DEVICE allocator_type
    get_allocator() const noexcept;

    /**
     * \brief Checks whether the atomic operations are lock-free
     * \return True if the operations are lock-free, false otherwise
     */
    STDGPU_HOST_DEVICE bool
    is_lock_free() const noexcept;

    /**
     * \brief Atomically loads and returns the current value of the atomic object
     * \param[in] order The memory order
     * \return The current value of this object
     */
    STDGPU_HOST_DEVICE T
    load(const memory_order order = memory_order_seq_cst) const;

    /**
     * \brief Atomically loads and returns the current value of the atomic object
     * \return The current value of this object
     */
    STDGPU_HOST_DEVICE
    operator T() const; // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Atomically replaces the current value with desired one
     * \param[in] desired The value to store to the atomic object
     * \param[in] order The memory order
     */
    STDGPU_HOST_DEVICE void
    store(const T desired, const memory_order order = memory_order_seq_cst);

    /**
     * \brief Atomically replaces the current value with desired one
     * \param[in] desired The value to store to the atomic object
     * \return The desired value
     */
    STDGPU_HOST_DEVICE T // NOLINT(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    operator=(const T desired);

    /**
     * \brief Atomically exchanges the current value with the given value
     * \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return The old value
     */
    STDGPU_DEVICE_ONLY T
    exchange(const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case
     * the both values are equal \param[in] expected A reference to the value to expect in the atomic object, will be
     * updated with old value if it has not been changed \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return True if the value has been changed to desired, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    compare_exchange_weak(T& expected, const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case
     * the both values are equal \param[in] expected A reference to the value to expect in the atomic object, will be
     * updated with old value if it has not been changed \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return True if the value has been changed to desired, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    compare_exchange_strong(T& expected, const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the addition of the stored value and the given argument
     * \param[in] arg The other argument of addition
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_add(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the subtraction of the stored value and the given argument
     * \param[in] arg The other argument of subtraction
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_sub(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
     * \param[in] arg The other argument of bitwise AND
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_and(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
     * \param[in] arg The other argument of bitwise OR
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_or(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
     * \param[in] arg The other argument of bitwise XOR
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_xor(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the minimum of the stored value and the given argument
     * \param[in] arg The other argument of minimum
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_min(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the maximum of the stored value and the given argument
     * \param[in] arg The other argument of maximum
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_max(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the incrementation of the value and modulus with arg
     * \param[in] arg The other argument of modulus
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_same_v<T, unsigned int>)>
    STDGPU_DEVICE_ONLY T
    fetch_inc_mod(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the decrementation of the value and modulus with arg
     * \param[in] arg The other argument of modulus
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_same_v<T, unsigned int>)>
    STDGPU_DEVICE_ONLY T
    fetch_dec_mod(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically increments the current value. Equivalent to fetch_add(1) + 1
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator++() noexcept;

    /**
     * \brief Atomically increments the current value. Equivalent to fetch_add(1)
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator++(int) noexcept;

    /**
     * \brief Atomically decrements the current value. Equivalent to fetch_sub(1) - 1
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator--() noexcept;

    /**
     * \brief Atomically decrements the current value. Equivalent to fetch_sub(1)
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator--(int) noexcept;

    /**
     * \brief Computes the atomic addition with the argument. Equivalent to fetch_add(arg) + arg
     * \param[in] arg The other argument of addition
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator+=(const T arg) noexcept;

    /**
     * \brief Computes the atomic subtraction with the argument. Equivalent to fetch_sub(arg) + arg
     * \param[in] arg The other argument of subtraction
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator-=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise AND with the argument. Equivalent to fetch_and(arg) & arg
     * \param[in] arg The other argument of bitwise AND
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator&=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise OR with the argument. Equivalent to fetch_or(arg) | arg
     * \param[in] arg The other argument of bitwise OR
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator|=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise XOR with the argument. Equivalent to fetch_xor(arg) ^ arg
     * \param[in] arg The other argument of bitwise XOR
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator^=(const T arg) noexcept;

private:
    explicit atomic(const Allocator& allocator) noexcept;

    atomic_ref<T> _value_ref;
    allocator_type _allocator = {};
};

using atomic_int = atomic<int>;                       /**< atomic<int> */
using atomic_uint = atomic<unsigned int>;             /**< atomic<unsigned int> */
using atomic_ullong = atomic<unsigned long long int>; /**< atomic<unsigned long long int> */

/**
 * \ingroup atomic
 * \brief A class to model a atomic reference to an object of type T on the GPU
 * \tparam T The type of the atomically managed object
 *
 * Supported types:
 *  - unsigned int
 *  - int
 *  - unsigned long long int
 *  - float (experimental)
 *
 * Differences to std::atomic_ref:
 *  - Is CopyAssignable
 *  - All operations (including load() and store()) may follow stricter ordering than requested
 *  - Additional min and max functions for all supported integer and floating point types
 *  - Additional increment/decrement + modulo functions for unsigned int
 */
template <typename T>
class atomic_ref
{
public:
    static_assert(std::is_same_v<T, unsigned int> || std::is_same_v<T, int> ||
                          std::is_same_v<T, unsigned long long int> || std::is_same_v<T, float>,
                  "stdgpu::atomic_ref: No support for type T");

    using value_type = T;               /**< T */
    using difference_type = value_type; /**< value_type */

    /**
     * \brief Deleted constructor
     */
    STDGPU_HOST_DEVICE
    atomic_ref() = delete;

    /**
     * \brief Constructor
     * \param[in] obj A reference to the object
     */
    STDGPU_HOST_DEVICE
    explicit atomic_ref(T& obj) noexcept;

    /**
     * \brief Checks whether the atomic operations are lock-free
     * \return True if the operations are lock-free, false otherwise
     */
    STDGPU_HOST_DEVICE bool
    is_lock_free() const noexcept;

    /**
     * \brief Loads and returns the current value of the atomic object
     * \param[in] order The memory order
     * \return The current value of this object
     */
    STDGPU_HOST_DEVICE T
    load(const memory_order order = memory_order_seq_cst) const;

    /**
     * \brief Loads and returns the current value of the atomic object
     * \return The current value of this object
     * \note Equivalent to load()
     */
    STDGPU_HOST_DEVICE
    operator T() const; // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Replaces the current value with desired
     * \param[in] desired The value to store to the atomic object
     * \param[in] order The memory order
     */
    STDGPU_HOST_DEVICE void
    store(const T desired, const memory_order order = memory_order_seq_cst);

    /**
     * \brief Replaces the current value with desired
     * \param[in] desired The value to store to the atomic object
     * \return The desired value
     * \note Equivalent to store()
     */
    STDGPU_HOST_DEVICE T // NOLINT(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    operator=(const T desired);

    /**
     * \brief Atomically exchanges the current value with the given value
     * \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return The old value
     */
    STDGPU_DEVICE_ONLY T
    exchange(const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case
     * the both values are equal \param[in] expected A reference to the value to expect in the atomic object, will be
     * updated with old value if it has not been changed \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return True if the value has been changed to desired, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    compare_exchange_weak(T& expected, const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case
     * the both values are equal \param[in] expected A reference to the value to expect in the atomic object, will be
     * updated with old value if it has not been changed \param[in] desired The value to exchange with the atomic object
     * \param[in] order The memory order
     * \return True if the value has been changed to desired, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    compare_exchange_strong(T& expected, const T desired, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the addition of the stored value and the given argument
     * \param[in] arg The other argument of addition
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_add(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the subtraction of the stored value and the given argument
     * \param[in] arg The other argument of subtraction
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_sub(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
     * \param[in] arg The other argument of bitwise AND
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_and(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
     * \param[in] arg The other argument of bitwise OR
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_or(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
     * \param[in] arg The other argument of bitwise XOR
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_xor(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the minimum of the stored value and the given argument
     * \param[in] arg The other argument of minimum
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_min(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the maximum of the stored value and the given argument
     * \param[in] arg The other argument of maximum
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    fetch_max(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the incrementation of the value and modulus with arg
     * \param[in] arg The other argument of modulus
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_same_v<T, unsigned int>)>
    STDGPU_DEVICE_ONLY T
    fetch_inc_mod(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically computes and stores the decrementation of the value and modulus with arg
     * \param[in] arg The other argument of modulus
     * \param[in] order The memory order
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_same_v<T, unsigned int>)>
    STDGPU_DEVICE_ONLY T
    fetch_dec_mod(const T arg, const memory_order order = memory_order_seq_cst) noexcept;

    /**
     * \brief Atomically increments the current value. Equivalent to fetch_add(1) + 1
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator++() noexcept;

    /**
     * \brief Atomically increments the current value. Equivalent to fetch_add(1)
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator++(int) noexcept;

    /**
     * \brief Atomically decrements the current value. Equivalent to fetch_sub(1) - 1
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator--() noexcept;

    /**
     * \brief Atomically decrements the current value. Equivalent to fetch_sub(1)
     * \return The old value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator--(int) noexcept;

    /**
     * \brief Computes the atomic addition with the argument. Equivalent to fetch_add(arg) + arg
     * \param[in] arg The other argument of addition
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator+=(const T arg) noexcept;

    /**
     * \brief Computes the atomic subtraction with the argument. Equivalent to fetch_sub(arg) + arg
     * \param[in] arg The other argument of subtraction
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T> || std::is_floating_point_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator-=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise AND with the argument. Equivalent to fetch_and(arg) & arg
     * \param[in] arg The other argument of bitwise AND
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator&=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise OR with the argument. Equivalent to fetch_or(arg) | arg
     * \param[in] arg The other argument of bitwise OR
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator|=(const T arg) noexcept;

    /**
     * \brief Computes the atomic bitwise XOR with the argument. Equivalent to fetch_xor(arg) ^ arg
     * \param[in] arg The other argument of bitwise XOR
     * \return The new value
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_integral_v<T>)>
    STDGPU_DEVICE_ONLY T
    operator^=(const T arg) noexcept;

private:
    template <typename T2, typename Allocator>
    friend class atomic;

    STDGPU_HOST_DEVICE
    explicit atomic_ref(T* value) noexcept;

    T* _value = nullptr;
};

/**
 * \ingroup atomic
 * \brief Checks whether the atomic operations are lock-free
 * \param[in] obj The atomic object
 * \return True if the operations are lock-free, false otherwise
 */
template <typename T, typename Allocator>
STDGPU_HOST_DEVICE bool
atomic_is_lock_free(const atomic<T, Allocator>* obj) noexcept;

/**
 * \ingroup atomic
 * \brief Loads and returns the current value of the atomic object
 * \param[in] obj The atomic object
 * \return The current value of this object
 */
template <typename T, typename Allocator>
STDGPU_HOST_DEVICE T
atomic_load(const atomic<T, Allocator>* obj) noexcept;

/**
 * \ingroup atomic
 * \brief Loads and returns the current value of the atomic object
 * \param[in] obj The atomic object
 * \param[in] order The memory order
 * \return The current value of this object
 */
template <typename T, typename Allocator>
STDGPU_HOST_DEVICE T
atomic_load_explicit(const atomic<T, Allocator>* obj, const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Replaces the current value with desired
 * \param[in] desired The value to store to the atomic object
 * \param[in] obj The atomic object
 */
template <typename T, typename Allocator>
STDGPU_HOST_DEVICE void
atomic_store(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::value_type desired) noexcept;

/**
 * \ingroup atomic
 * \brief Replaces the current value with desired
 * \param[in] obj The atomic object
 * \param[in] desired The value to store to the atomic object
 * \param[in] order The memory order
 */
template <typename T, typename Allocator>
STDGPU_HOST_DEVICE void
atomic_store_explicit(atomic<T, Allocator>* obj,
                      const typename atomic<T, Allocator>::value_type desired,
                      const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically exchanges the current value with the given value
 * \param[in] obj The atomic object
 * \param[in] desired The value to exchange with the atomic object
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_exchange(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::value_type desired) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically exchanges the current value with the given value
 * \param[in] obj The atomic object
 * \param[in] desired The value to exchange with the atomic object
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_exchange_explicit(atomic<T, Allocator>* obj,
                         const typename atomic<T, Allocator>::value_type desired,
                         const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the
 * both values are equal \param[in] obj The atomic object \param[in] expected A pointer to the value to expect in the
 * atomic object, will be updated with old value if it has not been changed \param[in] desired The value to exchange
 * with the atomic object \return True if the value has been changed to desired, false otherwise
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY bool
atomic_compare_exchange_weak(atomic<T, Allocator>* obj,
                             typename atomic<T, Allocator>::value_type* expected,
                             const typename atomic<T, Allocator>::value_type desired) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the
 * both values are equal \param[in] obj The atomic object \param[in] expected A pointer to the value to expect in the
 * atomic object, will be updated with old value if it has not been changed \param[in] desired The value to exchange
 * with the atomic object \return True if the value has been changed to desired, false otherwise
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY bool
atomic_compare_exchange_strong(atomic<T, Allocator>* obj,
                               typename atomic<T, Allocator>::value_type* expected,
                               const typename atomic<T, Allocator>::value_type desired) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the addition of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of addition
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_add(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::difference_type arg) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the addition of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of addition
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_add_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the subtraction of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of subtraction
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::difference_type arg) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the subtraction of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of subtraction
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_sub_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the addition of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of addition
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_and(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::difference_type arg) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of bitwise AND
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_and_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of bitwise OR
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_or(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::difference_type arg) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of bitwise OR
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_or_explicit(atomic<T, Allocator>* obj,
                         const typename atomic<T, Allocator>::difference_type arg,
                         const memory_order order) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of bitwise XOR
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor(atomic<T, Allocator>* obj, const typename atomic<T, Allocator>::difference_type arg) noexcept;

/**
 * \ingroup atomic
 * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
 * \param[in] obj The atomic object
 * \param[in] arg The other argument of bitwise XOR
 * \param[in] order The memory order
 * \return The old value
 */
template <typename T, typename Allocator>
STDGPU_DEVICE_ONLY T
atomic_fetch_xor_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order) noexcept;

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/atomic_detail.cuh>

#endif // STDGPU_ATOMIC_H
