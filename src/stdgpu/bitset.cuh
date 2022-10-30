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

#include <stdgpu/impl/platform_check.h>

/**
 * \addtogroup bitset bitset
 * \ingroup data_structures
 * @{
 */

/**
 * \file stdgpu/bitset.cuh
 */

#include <limits>
#include <type_traits>

#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>

///////////////////////////////////////////////////////////

#include <stdgpu/bitset_fwd>

///////////////////////////////////////////////////////////

namespace stdgpu
{

/**
 * \ingroup bitset
 * \brief A class to model a bitset on the GPU
 * \tparam Block The internal bit block type
 * \tparam Allocator The allocator type
 *
 * Differences to std::bitset:
 *  - Manual allocation and destruction of container required
 *  - set(), reset() and flip() return old state rather than reference to itself
 */
template <typename Block, typename Allocator>
class bitset
{
public:
    /**
     * \ingroup bitset
     * \brief A proxy class for a reference to a bit
     *
     * Differences to std::bitset::reference:
     *  - operator= and flip return old state rather than reference to itself
     */
    class reference
    {
    public:
        /**
         * \brief Deleted constructor
         */
        STDGPU_HOST_DEVICE
        reference() = delete;

        /**
         * \brief Default destructor
         */
        ~reference() noexcept = default;

        /**
         * \brief Default copy constructor
         * \param[in] x The reference object to copy
         */
        STDGPU_HOST_DEVICE
        reference(const reference& x) noexcept = default;

        /**
         * \brief Performs atomic assignment of a bit value
         * \param[in] x A bit value to assign
         * \return The old value of the bit
         */
        STDGPU_DEVICE_ONLY bool // NOLINT(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
        operator=(bool x) noexcept;

        /**
         * \brief Performs atomic assignment of a bit value
         * \param[in] x The reference object to assign
         * \return The old value of the bit
         */
        STDGPU_DEVICE_ONLY bool // NOLINT(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
        operator=(const reference& x) noexcept;

        /**
         * \brief Deleted move constructor
         */
        reference(reference&&) = delete;

        /**
         * \brief Deleted move assignment operator
         */
        reference&
        operator=(reference&&) = delete;

        /**
         * \brief Returns the value of the bit
         * \return The value of the bit
         */
        STDGPU_DEVICE_ONLY
        operator bool() const noexcept; // NOLINT(hicpp-explicit-conversions)

        /**
         * \brief Returns the inverse of the value of the bit
         * \return The inverse of the value of the bit
         */
        STDGPU_DEVICE_ONLY bool
        operator~() const noexcept;

        /**
         * \brief Flips the bit atomically
         * \return The old value of the bit
         */
        STDGPU_DEVICE_ONLY bool
        flip() noexcept;

    private:
        using block_type = Block; /**< The type of the stored bit blocks, must be the same as for bitset */

        static_assert(std::is_same_v<block_type, unsigned int> || std::is_same_v<block_type, unsigned long long int>,
                      "stdgpu::bitset::reference: block_type not supported");

        friend bitset;

        STDGPU_HOST_DEVICE
        reference(block_type* bit_block, const index_t bit_n);

        static STDGPU_DEVICE_ONLY bool
        bit(block_type bits, const index_t n) noexcept;

        static constexpr index_t _bits_per_block = std::numeric_limits<block_type>::digits;

        block_type* _bit_block = nullptr;
        index_t _bit_n = -1;
    };

    static_assert(std::is_same_v<Block, unsigned int> || std::is_same_v<Block, unsigned long long int>,
                  "stdgpu::bitset: Block not supported");

    using block_type = Block;         /**< Block */
    using allocator_type = Allocator; /**< Allocator */

    /**
     * \brief Creates an object of this class on the GPU (device)
     * \param[in] size The size of this object
     * \param[in] allocator The allocator instance to use
     * \return A newly created object of this class allocated on the GPU (device)
     */
    static bitset
    createDeviceObject(const index_t& size, const Allocator& allocator = Allocator());

    /**
     * \brief Destroys the given object of this class on the GPU (device)
     * \param[in] device_object The object allocated on the GPU (device)
     */
    static void
    destroyDeviceObject(bitset& device_object);

    /**
     * \brief Empty constructor
     */
    bitset() noexcept = default;

    /**
     * \brief Returns the container allocator
     * \return The container allocator
     */
    STDGPU_HOST_DEVICE allocator_type
    get_allocator() const noexcept;

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
    STDGPU_DEVICE_ONLY bool
    set(const index_t n, const bool value = true);

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
    STDGPU_DEVICE_ONLY bool
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
    STDGPU_DEVICE_ONLY bool
    flip(const index_t n);

    /**
     * \brief Returns the bit at the given position
     * \param[in] n The position
     * \return The bit at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY bool
    operator[](const index_t n) const;

    /**
     * \brief Returns a reference object to the bit at the given position
     * \param[in] n The position
     * \return A reference object to the bit at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY reference
    operator[](const index_t n);

    /**
     * \brief Returns the bit at the given position
     * \param[in] n The position
     * \return The bit at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY bool
    test(const index_t n) const;

    /**
     * \brief Checks if this object is empty
     * \return True if this object is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const noexcept;

    /**
     * \brief The size
     * \return The size of the object
     */
    STDGPU_HOST_DEVICE index_t
    size() const noexcept;

    /**
     * \brief The number of set bits
     * \return The number of set bits
     */
    index_t
    count() const;

    /**
     * \brief Checks if all bits are set
     * \return True if all bits are set, false otherwise
     */
    bool
    all() const;

    /**
     * \brief Checks if any bits are set
     * \return True if any bits are set, false otherwise
     */
    bool
    any() const;

    /**
     * \brief Checks if none of the bits are set
     * \return True if none of the bits are set, false otherwise
     */
    bool
    none() const;

private:
    explicit bitset(const Allocator& allocator) noexcept;

    static index_t
    number_bit_blocks(const index_t size) noexcept;

    static constexpr index_t _bits_per_block = std::numeric_limits<block_type>::digits;

    block_type* _bit_blocks = nullptr;
    index_t _size = 0;
    allocator_type _allocator = {};
};

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/bitset_detail.cuh>

#endif // STDGPU_BITSET_H
