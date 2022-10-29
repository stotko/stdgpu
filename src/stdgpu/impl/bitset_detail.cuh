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

#ifndef STDGPU_BITSET_DETAIL_H
#define STDGPU_BITSET_DETAIL_H

#include <limits>

#include <stdgpu/algorithm.h>
#include <stdgpu/atomic.cuh>
#include <stdgpu/bit.h>
#include <stdgpu/contract.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/numeric.h>

namespace stdgpu
{

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE
bitset<Block, Allocator>::reference::reference(bitset<Block, Allocator>::reference::block_type* bit_block,
                                               const index_t bit_n)
  : _bit_block(bit_block)
  , _bit_n(bit_n)
{
    STDGPU_EXPECTS(0 <= bit_n);
    STDGPU_EXPECTS(bit_n < _bits_per_block);
}

// NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
template <typename Block, typename Allocator>
// NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::operator=(bool x) noexcept
{
    block_type set_pattern = static_cast<block_type>(1) << static_cast<block_type>(_bit_n);
    block_type reset_pattern = ~set_pattern;

    block_type old;
    atomic_ref<block_type> bit_block(*_bit_block);
    if (x)
    {
        old = bit_block.fetch_or(set_pattern);
    }
    else
    {
        old = bit_block.fetch_and(reset_pattern);
    }

    return bit(old, _bit_n);
}

// NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
template <typename Block, typename Allocator>
// NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
inline STDGPU_DEVICE_ONLY bool
// NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
bitset<Block, Allocator>::reference::operator=(const reference& x) noexcept
{
    return operator=(static_cast<bool>(x));
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bitset<Block, Allocator>::reference::operator bool() const noexcept
{
    atomic_ref<block_type> bit_block(*_bit_block);
    return bit(bit_block.load(), _bit_n);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::operator~() const noexcept
{
    return !operator bool();
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::flip() noexcept
{
    block_type flip_pattern = static_cast<block_type>(1) << static_cast<block_type>(_bit_n);

    atomic_ref<block_type> bit_block(*_bit_block);
    block_type old = bit_block.fetch_xor(flip_pattern);

    return bit(old, _bit_n);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::bit(bitset<Block, Allocator>::reference::block_type bits, const index_t n) noexcept
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < _bits_per_block);

    return ((bits & (static_cast<block_type>(1) << static_cast<block_type>(n))) != 0);
}

namespace detail
{

inline index_t
div_up(const index_t a, const index_t b) noexcept
{
    STDGPU_EXPECTS(a >= 0);
    STDGPU_EXPECTS(b > 0);

    index_t result = (a % b != 0) ? (a / b + 1) : (a / b);

    STDGPU_ENSURES(result * b >= a);

    return result;
}

template <typename Block>
class count_block_bits
{
public:
    inline count_block_bits(Block* bit_blocks, const index_t size)
      : _bit_blocks(bit_blocks)
      , _size(size)
    {
    }

    inline STDGPU_HOST_DEVICE index_t
    operator()(const index_t i) const
    {
        return static_cast<index_t>(popcount(block_mask(i) & _bit_blocks[i]));
    }

private:
    inline STDGPU_HOST_DEVICE Block
    block_mask(const index_t i) const
    {
        index_t remaining_bits = _size - i * _bits_per_block;
        return (remaining_bits >= _bits_per_block)
                       ? ~static_cast<Block>(0)
                       : (static_cast<Block>(1) << static_cast<Block>(remaining_bits)) - static_cast<Block>(1);
    }

    static constexpr index_t _bits_per_block = std::numeric_limits<Block>::digits;

    Block* _bit_blocks;
    index_t _size;
};

template <typename Block>
class flip_bits
{
public:
    inline explicit flip_bits(Block* bit_blocks)
      : _bit_blocks(bit_blocks)
    {
    }

    inline STDGPU_HOST_DEVICE void
    operator()(const index_t i)
    {
        _bit_blocks[i] = ~_bit_blocks[i];
    }

private:
    Block* _bit_blocks;
};

} // namespace detail

template <typename Block, typename Allocator>
inline bitset<Block, Allocator>
bitset<Block, Allocator>::createDeviceObject(const index_t& size, const Allocator& allocator)
{
    bitset<Block, Allocator> result(allocator);
    result._bit_blocks = createDeviceArray<block_type, Allocator>(result._allocator,
                                                                  number_bit_blocks(size),
                                                                  static_cast<block_type>(0));
    result._size = size;

    return result;
}

template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::destroyDeviceObject(bitset<Block, Allocator>& device_object)
{
    destroyDeviceArray<block_type, Allocator>(device_object._allocator, device_object._bit_blocks);
    device_object._bit_blocks = nullptr;
    device_object._size = 0;
}

template <typename Block, typename Allocator>
inline bitset<Block, Allocator>::bitset(const Allocator& allocator) noexcept
  : _allocator(allocator)
{
}

template <typename Block, typename Allocator>
inline index_t
bitset<Block, Allocator>::number_bit_blocks(const index_t size) noexcept
{
    return detail::div_up(size, _bits_per_block);
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE typename bitset<Block, Allocator>::allocator_type
bitset<Block, Allocator>::get_allocator() const noexcept
{
    return _allocator;
}

template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::set()
{
    fill(execution::device, device_begin(_bit_blocks), device_end(_bit_blocks), ~block_type(0));

    STDGPU_ENSURES(count() == size());
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::set(const index_t n, const bool value)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n) = value;
}

template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::reset()
{
    fill(execution::device, device_begin(_bit_blocks), device_end(_bit_blocks), block_type(0));

    STDGPU_ENSURES(count() == 0);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reset(const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return set(n, false);
}

template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::flip()
{
    for_each_index(execution::device, number_bit_blocks(size()), detail::flip_bits<Block>(_bit_blocks));
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::flip(const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n).flip();
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::operator[](const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    index_t block_n = n / _bits_per_block;
    index_t bit_n = static_cast<index_t>(
            bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

    return reference(_bit_blocks + block_n, bit_n);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY typename bitset<Block, Allocator>::reference
bitset<Block, Allocator>::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    index_t block_n = n / _bits_per_block;
    index_t bit_n = static_cast<index_t>(
            bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

    return reference(_bit_blocks + block_n, bit_n);
}

template <typename Block, typename Allocator>

inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::test(const index_t n) const
{
    return operator[](n);
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE bool
bitset<Block, Allocator>::empty() const noexcept
{
    return (size() == 0);
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
bitset<Block, Allocator>::size() const noexcept
{
    return _size;
}

template <typename Block, typename Allocator>
inline index_t
bitset<Block, Allocator>::count() const
{
    if (size() == 0)
    {
        return 0;
    }

    return transform_reduce_index(execution::device,
                                  number_bit_blocks(size()),
                                  0,
                                  plus<index_t>(),
                                  detail::count_block_bits<Block>(_bit_blocks, size()));
}

template <typename Block, typename Allocator>
inline bool
bitset<Block, Allocator>::all() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == size();
}

template <typename Block, typename Allocator>
inline bool
bitset<Block, Allocator>::any() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() > 0;
}

template <typename Block, typename Allocator>
inline bool
bitset<Block, Allocator>::none() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == 0;
}

} // namespace stdgpu

#endif // STDGPU_BITSET_DETAIL_H
