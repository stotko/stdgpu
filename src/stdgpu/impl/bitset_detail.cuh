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
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>

#include <stdgpu/atomic.cuh>
#include <stdgpu/bit.h>
#include <stdgpu/contract.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



namespace stdgpu
{

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE
bitset<Block, Allocator>::reference::reference(bitset<Block, Allocator>::reference::block_type* bit_block,
                                               const index_t bit_n)
    : _bit_block(bit_block),
      _bit_n(bit_n)
{
    STDGPU_EXPECTS(0 <= bit_n);
    STDGPU_EXPECTS(bit_n < _bits_per_block);
}


template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE
bitset<Block, Allocator>::reference::reference(const bitset<Block, Allocator>::reference& x) //NOLINT(hicpp-use-equals-default,modernize-use-equals-default)
    : _bit_block(x._bit_block),
      _bit_n(x._bit_n)
{

}


template <typename Block, typename Allocator> //NOLINT(misc-unconventional-assign-operator)
inline STDGPU_DEVICE_ONLY bool //NOLINT(misc-unconventional-assign-operator)
bitset<Block, Allocator>::reference::operator=(bool x)
{
    block_type set_pattern = static_cast<block_type>(1) << static_cast<block_type>(_bit_n);
    block_type reset_pattern = ~set_pattern;

    block_type old;
    stdgpu::atomic_ref<block_type> bit_block(*_bit_block);
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


template <typename Block, typename Allocator> //NOLINT(misc-unconventional-assign-operator)
inline STDGPU_DEVICE_ONLY bool //NOLINT(misc-unconventional-assign-operator)
bitset<Block, Allocator>::reference::operator=(const reference& x) //NOLINT(bugprone-unhandled-self-assignment)
{
    return operator=(static_cast<bool>(x));
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY
bitset<Block, Allocator>::reference::operator bool() const
{
    stdgpu::atomic_ref<block_type> bit_block(*_bit_block);
    return bit(bit_block.load(), _bit_n);
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::operator~() const
{
    return !operator bool();
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::flip()
{
    block_type flip_pattern = static_cast<block_type>(1) << static_cast<block_type>(_bit_n);

    stdgpu::atomic_ref<block_type> bit_block(*_bit_block);
    block_type old = bit_block.fetch_xor(flip_pattern);

    return bit(old, _bit_n);
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::reference::bit(bitset<Block, Allocator>::reference::block_type bits,
                       const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < _bits_per_block);

    return ((bits & (static_cast<block_type>(1) << static_cast<block_type>(n))) != 0);
}


namespace detail
{

inline index_t
div_up(const index_t a,
       const index_t b)
{
    STDGPU_EXPECTS(a >= 0);
    STDGPU_EXPECTS(b > 0);

    index_t result = (a % b != 0) ? (a / b + 1) : (a / b);

    STDGPU_ENSURES(result * b >= a);

    return result;
}

template <typename T>
struct count_block_bits
{
    inline STDGPU_HOST_DEVICE index_t
    operator()(const T pattern) const
    {
        return static_cast<index_t>(popcount(pattern));
    }
};

template <typename Block, typename Allocator>
class count_bits
{
    public:
        inline
        explicit count_bits(const bitset<Block, Allocator>& bits)
            : _bits(bits)
        {

        }

        inline STDGPU_DEVICE_ONLY index_t
        operator()(const index_t i)
        {
            return static_cast<index_t>(_bits.test(i));
        }

    private:
        bitset<Block, Allocator> _bits;
};

} // namespace detail



template <typename Block, typename Allocator>
inline bitset<Block, Allocator>
bitset<Block, Allocator>::createDeviceObject(const index_t& size,
                                             const Allocator& allocator)
{
    bitset<Block, Allocator> result(allocator);
    result._bit_blocks          = createDeviceArray<block_type, Allocator>(result._allocator, number_bit_blocks(size), static_cast<block_type>(0));
    result._size                = size;

    return result;
}


template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::destroyDeviceObject(bitset<Block, Allocator>& device_object)
{
    destroyDeviceArray<block_type, Allocator>(device_object._allocator, device_object._bit_blocks);
    device_object._bit_blocks   = nullptr;
    device_object._size         = 0;
}


template <typename Block, typename Allocator>
inline
bitset<Block, Allocator>::bitset(const Allocator& allocator)
    : _allocator(allocator)
{

}


template <typename Block, typename Allocator>
inline index_t
bitset<Block, Allocator>::number_bit_blocks(const index_t size)
{
    return detail::div_up(size, _bits_per_block);
}


template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE typename bitset<Block, Allocator>::allocator_type
bitset<Block, Allocator>::get_allocator() const
{
    return _allocator;
}


template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::set()
{
    thrust::fill(device_begin(_bit_blocks), device_end(_bit_blocks),
                 ~block_type(0));

    STDGPU_ENSURES(count() == size());
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
bitset<Block, Allocator>::set(const index_t n,
            const bool value)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n) = value;
}


template <typename Block, typename Allocator>
inline void
bitset<Block, Allocator>::reset()
{
    thrust::fill(device_begin(_bit_blocks), device_end(_bit_blocks),
                 block_type(0));

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
    thrust::transform(device_begin(_bit_blocks), device_end(_bit_blocks),
                      device_begin(_bit_blocks),
                      bit_not<block_type>());
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
    index_t bit_n = static_cast<index_t>(bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

    return reference(_bit_blocks + block_n, bit_n);
}


template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY typename bitset<Block, Allocator>::reference
bitset<Block, Allocator>::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    index_t block_n = n / _bits_per_block;
    index_t bit_n = static_cast<index_t>(bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

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
bitset<Block, Allocator>::empty() const
{
    return (size() == 0);
}


template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
bitset<Block, Allocator>::size() const
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

    index_t full_blocks_count = thrust::transform_reduce(device_begin(_bit_blocks), device_end(_bit_blocks) - 1,
                                                         detail::count_block_bits<block_type>(),
                                                         0,
                                                         thrust::plus<index_t>());

    index_t last_block_count = thrust::transform_reduce(thrust::counting_iterator<index_t>((number_bit_blocks(size()) - 1) * _bits_per_block), thrust::counting_iterator<index_t>(size()),
                                                        detail::count_bits<Block, Allocator>(*this),
                                                        0,
                                                        thrust::plus<index_t>());

    return full_blocks_count + last_block_count;
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
