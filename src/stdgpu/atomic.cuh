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

/**
 * \file stdgpu/atomic.cuh
 */

#include <cstddef>
#include <type_traits>

#include <stdgpu/platform.h>



///////////////////////////////////////////////////////////


#include <stdgpu/atomic_fwd>


///////////////////////////////////////////////////////////



/**
 * \brief Atomically computes the difference of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ unsigned long long int
atomicSub(unsigned long long int* address,
          const unsigned long long int value);

/**
 * \brief Atomically computes the difference of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicSub(float* address,
          const float value);


/**
 * \brief Atomically computes the minimum of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicMin(float* address,
          const float value);


/**
 * \brief Atomically computes the maximum of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicMax(float* address,
          const float value);


/**
 * \brief Atomically computes the minimum of the two positive values
 * \param[in] address A pointer to a positive value
 * \param[in] value A positive value
 * \return The old value at the given address
 */
__device__ float
atomicMinPositive(float* address,
                  const float value);


/**
 * \brief Atomically computes the maximum of the two positive values
 * \param[in] address A pointer to a positive value
 * \param[in] value A positive value
 * \return The old value at the given address
 */
__device__ float
atomicMaxPositive(float* address,
                  const float value);



namespace stdgpu
{

/**
 * \brief A class to model an atomic object of type T on the GPU
 * \tparam T The type of the atomically managed object
 *
 * Supported types:
 *  - unsigned int
 *  - int
 *  - unsigned long long int
 *  - float (experimental)
 *
 * Differences to std::atomic:
 *  - Atomics must be modeled as containers since threads have to operate on the exact same object (which also requires copy and move constructors)
 *  - Manual allocation and destruction of container required
 *  - load and store are not atomically safe
 *  - Additional min and max functions for all supported integer and floating point types
 *  - Additional increment/decrement + modulo functions for unsigned int
 */
template <typename T>
class atomic
{
    public:
        static_assert(std::is_same<T, unsigned int>::value ||
                      std::is_same<T, int>::value ||
                      std::is_same<T, unsigned long long int>::value ||
                      std::is_same<T, float>::value,
                      "stdgpu::atomic: No support for type T");

        using value_type = T;                   /**< T */
        using difference_type = value_type;     /**< value_type */


        /**
         * \brief Creates an object of this class on the GPU (device)
         * \return A newly created object of this class allocated on the GPU (device)
         * \note The size is implictly set to 1 (and not needed as a parameter) as the object only manages a single value
         */
        static atomic
        createDeviceObject();

        /**
         * \brief Destroys the given object of this class on the GPU (device)
         * \param[in] device_object The object allocated on the GPU (device)
         */
        static void
        destroyDeviceObject(atomic& device_object);


        /**
         * \brief Empty constructor
         */
        atomic();


        /**
         * \brief Loads and returns the current value of the atomic object
         * \return The current value of this object
         * \note This operation is not atomically safe
         */
        STDGPU_HOST_DEVICE T
        load() const;

        /**
         * \brief Replaces the current value with desired
         * \param[in] desired The value to store to the atomic object
         * \note This operation is not atomically safe
         */
        STDGPU_HOST_DEVICE void
        store(const T desired);


        /**
         * \brief Atomically exchanges the current value with the given value
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ T
        exchange(const T desired);


        /**
         * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the both values are equal
         * \param[in] expected The value to expect in the atomic object
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ bool
        compare_exchange_weak(T& expected,
                              const T desired);

        /**
         * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the both values are equal
         * \param[in] expected The value to expect in the atomic object
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ bool
        compare_exchange_strong(T& expected,
                                const T desired);


        /**
         * \brief Atomically computes and stores the addition of the stored value and the given argument
         * \param[in] arg The other argument of addition
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_add(const T arg);

        /**
         * \brief Atomically computes and stores the subtraction of the stored value and the given argument
         * \param[in] arg The other argument of subtraction
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_sub(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
         * \param[in] arg The other argument of bitwise AND
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_and(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
         * \param[in] arg The other argument of bitwise OR
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_or(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
         * \param[in] arg The other argument of bitwise XOR
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_xor(const T arg);


        /**
         * \brief Atomically computes and stores the minimum of the stored value and the given argument
         * \param[in] arg The other argument of minimum
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_min(const T arg);

        /**
         * \brief Atomically computes and stores the maximum of the stored value and the given argument
         * \param[in] arg The other argument of maximum
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_max(const T arg);

        /**
         * \brief Atomically computes and stores the incrementation of the value and modulus with arg
         * \param[in] arg The other argument of modulus
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_same<U, unsigned int>::value>>
        __device__ T
        fetch_inc_mod(const T arg);

        /**
         * \brief Atomically computes and stores the decrementation of the value and modulus with arg
         * \param[in] arg The other argument of modulus
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_same<U, unsigned int>::value>>
        __device__ T
        fetch_dec_mod(const T arg);


        /**
         * \brief Atomically increments the current value. Equivalent to fetch_add(1) + 1
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator++();

        /**
         * \brief Atomically increments the current value. Equivalent to fetch_add(1)
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator++(int);

        /**
         * \brief Atomically decrements the current value. Equivalent to fetch_sub(1) - 1
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator--();

        /**
         * \brief Atomically decrements the current value. Equivalent to fetch_sub(1)
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator--(int);


        /**
         * \brief Computes the atomic addition with the argument. Equivalent to fetch_add(arg) + arg
         * \param[in] arg The other argument of addition
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        operator+=(const T arg);

        /**
         * \brief Computes the atomic subtraction with the argument. Equivalent to fetch_sub(arg) + arg
         * \param[in] arg The other argument of subtraction
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        operator-=(const T arg);

        /**
         * \brief Computes the atomic bitwise AND with the argument. Equivalent to fetch_and(arg) & arg
         * \param[in] arg The other argument of bitwise AND
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator&=(const T arg);

        /**
         * \brief Computes the atomic bitwise OR with the argument. Equivalent to fetch_or(arg) | arg
         * \param[in] arg The other argument of bitwise OR
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator|=(const T arg);

        /**
         * \brief Computes the atomic bitwise XOR with the argument. Equivalent to fetch_xor(arg) ^ arg
         * \param[in] arg The other argument of bitwise XOR
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator^=(const T arg);

    private:
        explicit atomic(T* value);


        T* _value = nullptr;
        atomic_ref<T> _value_ref;
};


/**
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
 *  - load and store are not atomically safe
 *  - Additional min and max functions for all supported integer and floating point types
 *  - Additional increment/decrement + modulo functions for unsigned int
 */
template <typename T>
class atomic_ref
{
    public:
        static_assert(std::is_same<T, unsigned int>::value ||
                      std::is_same<T, int>::value ||
                      std::is_same<T, unsigned long long int>::value ||
                      std::is_same<T, float>::value,
                      "stdgpu::atomic_ref: No support for type T");

        using value_type = T;                   /**< T */
        using difference_type = value_type;     /**< value_type */


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
        explicit atomic_ref(T& obj);


        /**
         * \brief Loads and returns the current value of the atomic object
         * \return The current value of this object
         * \note This operation is not atomically safe
         */
        STDGPU_HOST_DEVICE T
        load() const;

        /**
         * \brief Replaces the current value with desired
         * \param[in] desired The value to store to the atomic object
         * \note This operation is not atomically safe
         */
        STDGPU_HOST_DEVICE void
        store(const T desired);


        /**
         * \brief Atomically exchanges the current value with the given value
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ T
        exchange(const T desired);


        /**
         * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the both values are equal
         * \param[in] expected The value to expect in the atomic object
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ bool
        compare_exchange_weak(T& expected,
                              const T desired);

        /**
         * \brief Atomically compares the current value with the given value and exchanges it with the desired one in case the both values are equal
         * \param[in] expected The value to expect in the atomic object
         * \param[in] desired The value to exchange with the atomic object
         * \return The old value
         */
        __device__ bool
        compare_exchange_strong(T& expected,
                                const T desired);


        /**
         * \brief Atomically computes and stores the addition of the stored value and the given argument
         * \param[in] arg The other argument of addition
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_add(const T arg);

        /**
         * \brief Atomically computes and stores the subtraction of the stored value and the given argument
         * \param[in] arg The other argument of subtraction
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_sub(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise AND of the stored value and the given argument
         * \param[in] arg The other argument of bitwise AND
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_and(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise OR of the stored value and the given argument
         * \param[in] arg The other argument of bitwise OR
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_or(const T arg);

        /**
         * \brief Atomically computes and stores the bitwise XOR of the stored value and the given argument
         * \param[in] arg The other argument of bitwise XOR
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        fetch_xor(const T arg);


        /**
         * \brief Atomically computes and stores the minimum of the stored value and the given argument
         * \param[in] arg The other argument of minimum
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_min(const T arg);

        /**
         * \brief Atomically computes and stores the maximum of the stored value and the given argument
         * \param[in] arg The other argument of maximum
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        fetch_max(const T arg);

        /**
         * \brief Atomically computes and stores the incrementation of the value and modulus with arg
         * \param[in] arg The other argument of modulus
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_same<U, unsigned int>::value>>
        __device__ T
        fetch_inc_mod(const T arg);

        /**
         * \brief Atomically computes and stores the decrementation of the value and modulus with arg
         * \param[in] arg The other argument of modulus
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_same<U, unsigned int>::value>>
        __device__ T
        fetch_dec_mod(const T arg);


        /**
         * \brief Atomically increments the current value. Equivalent to fetch_add(1) + 1
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator++();

        /**
         * \brief Atomically increments the current value. Equivalent to fetch_add(1)
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator++(int);

        /**
         * \brief Atomically decrements the current value. Equivalent to fetch_sub(1) - 1
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator--();

        /**
         * \brief Atomically decrements the current value. Equivalent to fetch_sub(1)
         * \return The old value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator--(int);


        /**
         * \brief Computes the atomic addition with the argument. Equivalent to fetch_add(arg) + arg
         * \param[in] arg The other argument of addition
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        operator+=(const T arg);

        /**
         * \brief Computes the atomic subtraction with the argument. Equivalent to fetch_sub(arg) + arg
         * \param[in] arg The other argument of subtraction
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value || std::is_floating_point<U>::value>>
        __device__ T
        operator-=(const T arg);

        /**
         * \brief Computes the atomic bitwise AND with the argument. Equivalent to fetch_and(arg) & arg
         * \param[in] arg The other argument of bitwise AND
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator&=(const T arg);

        /**
         * \brief Computes the atomic bitwise OR with the argument. Equivalent to fetch_or(arg) | arg
         * \param[in] arg The other argument of bitwise OR
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator|=(const T arg);

        /**
         * \brief Computes the atomic bitwise XOR with the argument. Equivalent to fetch_xor(arg) ^ arg
         * \param[in] arg The other argument of bitwise XOR
         * \return The new value
         */
        template <typename U = T, typename = std::enable_if_t<std::is_integral<U>::value>>
        __device__ T
        operator^=(const T arg);

    private:
        friend atomic<T>;

        STDGPU_HOST_DEVICE
        explicit atomic_ref(T* value);

        T* _value = nullptr;
};

} // namespace stdgpu



#include <stdgpu/impl/atomic_detail.cuh>



#endif // STDGPU_ATOMIC_H
