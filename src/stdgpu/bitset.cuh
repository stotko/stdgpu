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

#ifndef STDGPU_BITSET_H
#define STDGPU_BITSET_H

/**
 * \file stdgpu/bitset.cuh
 */

#include <limits>
#include <type_traits>

#include <stdgpu/attribute.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>



///////////////////////////////////////////////////////////


#include <stdgpu/bitset_fwd>


///////////////////////////////////////////////////////////



namespace stdgpu
{

/**
 * \brief A class to model a bitset on the GPU
 *
 * Differences to std::bitset:
 *  - Manual allocation and destruction of container required
 *  - set(), reset() and flip() return old state rather than reference to itself
 */
class bitset
{
    public:
        /**
         * \brief Creates an object of this class on the GPU (device)
         * \param[in] size The size of this object
         * \return A newly created object of this class allocated on the GPU (device)
         */
        static bitset
        createDeviceObject(const index_t& size);

        /**
         * \brief Destroys the given object of this class on the GPU (device)
         * \param[in] device_object The object allocated on the GPU (device)
         */
        static void
        destroyDeviceObject(bitset& device_object);


        /**
         * \brief Empty constructor
         */
        bitset() = default;

        /**
         * \brief Sets all bits
         * \post count() == size()
         */
        void
        set();

        /**
         * \brief Sets the bit at the given position
         * \param[in] n The position that should be set
         * \param[in] value The new value of the bit
         * \return The old value of the bit
         * \pre 0 <= n < size()
         */
        __device__ bool
        set(const index_t n,
            const bool value = true);

        /**
         * \brief Resets all bits
         * \post count() == 0
         */
        void
        reset();

        /**
         * \brief Resets the bit at the given position. Equivalent to : set(n, false)
         * \param[in] n The position that should be reset
         * \return The old value of the bit
         * \pre 0 <= n < size()
         */
        __device__ bool
        reset(const index_t n);

        /**
         * \brief Flips all bits
         */
        void
        flip();

        /**
         * \brief Flips the bit at the given position
         * \param[in] n The position that should be flipped
         * \return The old value of the bit
         * \pre 0 <= n < size()
         */
        __device__ bool
        flip(const index_t n);

        /**
         * \brief Returns the bit at the given position
         * \param[in] n The position
         * \return The bit at this position
         * \pre 0 <= n < size()
         */
        __device__ bool
        operator[](const index_t n) const;


        /**
         * \brief Checks if this object is empty
         * \return True if this object is empty, false otherwise
         */
        STDGPU_NODISCARD STDGPU_HOST_DEVICE bool
        empty() const;

        /**
         * \brief The size
         * \return The size of the object
         */
        STDGPU_HOST_DEVICE index_t
        size() const;

        /**
         * \brief The number of set bits
         * \return The number of set bits
         */
        index_t
        count() const;

    private:
        using block_type = unsigned int;        /**< The type of the stored bit blocks */

        static_assert(std::is_same<block_type, unsigned int>::value ||
                      std::is_same<block_type, unsigned long long int>::value,
                      "stdgpu::bitset: block_type not supported");


        block_type* _bit_blocks = nullptr;
        index_t _bits_per_block = std::numeric_limits<block_type>::digits;
        index_t _number_bit_blocks = 0;
        index_t _size = 0;
};

} // namespace stdgpu



#include <stdgpu/impl/bitset_detail.cuh>



#endif // STDGPU_BITSET_H
