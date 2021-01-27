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

inline STDGPU_HOST_DEVICE
bitset::reference::reference(bitset::reference::block_type* bit_block,
                             const index_t bit_n)
    : _bit_block(bit_block),
      _bit_n(bit_n)
{
    STDGPU_EXPECTS(0 <= bit_n);
    STDGPU_EXPECTS(bit_n < _bits_per_block);
}


inline STDGPU_HOST_DEVICE
bitset::reference::reference(const bitset::reference& x) //NOLINT(hicpp-use-equals-default,modernize-use-equals-default)
    : _bit_block(x._bit_block),
      _bit_n(x._bit_n)
{

}


inline STDGPU_DEVICE_ONLY bool //NOLINT(misc-unconventional-assign-operator)
bitset::reference::operator=(bool x)
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


inline STDGPU_DEVICE_ONLY bool //NOLINT(misc-unconventional-assign-operator)
bitset::reference::operator=(const reference& x)
{
    return operator=(static_cast<bool>(x));
}


inline STDGPU_DEVICE_ONLY
bitset::reference::operator bool() const
{
    stdgpu::atomic_ref<block_type> bit_block(*_bit_block);
    return bit(bit_block.load(), _bit_n);
}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::operator~() const
{
    return !operator bool();
}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::flip()
{
    block_type flip_pattern = static_cast<block_type>(1) << static_cast<block_type>(_bit_n);

    stdgpu::atomic_ref<block_type> bit_block(*_bit_block);
    block_type old = bit_block.fetch_xor(flip_pattern);

    return bit(old, _bit_n);
}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::bit(bitset::reference::block_type bits,
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

class count_bits
{
    public:
        inline
        explicit count_bits(const bitset& bits)
            : _bits(bits)
        {

        }

        inline STDGPU_DEVICE_ONLY index_t
        operator()(const index_t i)
        {
            return static_cast<index_t>(_bits.test(i));
        }

    private:
        bitset _bits;
};

} // namespace detail



inline bitset
bitset::createDeviceObject(const index_t& size)
{
    bitset result;
    result._number_bit_blocks   = detail::div_up(size, _bits_per_block);
    result._bit_blocks          = createDeviceArray<block_type>(result._number_bit_blocks, static_cast<block_type>(0));
    result._size                = size;

    return result;
}


inline void
bitset::destroyDeviceObject(bitset& device_object)
{
    destroyDeviceArray<block_type>(device_object._bit_blocks);
    device_object._bit_blocks   = nullptr;
    device_object._size         = 0;
}


inline void
bitset::set()
{
    thrust::fill(device_begin(_bit_blocks), device_end(_bit_blocks),
                 ~block_type(0));

    STDGPU_ENSURES(count() == size());
}


inline STDGPU_DEVICE_ONLY bool
bitset::set(const index_t n,
            const bool value)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n) = value;
}


inline void
bitset::reset()
{
    thrust::fill(device_begin(_bit_blocks), device_end(_bit_blocks),
                 block_type(0));

    STDGPU_ENSURES(count() == 0);
}


inline STDGPU_DEVICE_ONLY bool
bitset::reset(const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return set(n, false);
}


inline void
bitset::flip()
{
    thrust::transform(device_begin(_bit_blocks), device_end(_bit_blocks),
                      device_begin(_bit_blocks),
                      bit_not<block_type>());
}


inline STDGPU_DEVICE_ONLY bool
bitset::flip(const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n).flip();
}


inline STDGPU_DEVICE_ONLY bool
bitset::operator[](const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    index_t block_n = n / _bits_per_block;
    index_t bit_n = static_cast<index_t>(bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

    return reference(_bit_blocks + block_n, bit_n);
}


inline STDGPU_DEVICE_ONLY bitset::reference
bitset::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    index_t block_n = n / _bits_per_block;
    index_t bit_n = static_cast<index_t>(bit_mod<std::size_t>(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block)));

    return reference(_bit_blocks + block_n, bit_n);
}


inline STDGPU_DEVICE_ONLY bool
bitset::test(const index_t n) const
{
    return operator[](n);
}


inline STDGPU_HOST_DEVICE bool
bitset::empty() const
{
    return (size() == 0);
}


inline STDGPU_HOST_DEVICE index_t
bitset::size() const
{
    return _size;
}


inline index_t
bitset::count() const
{
    if (size() == 0)
    {
        return 0;
    }

    index_t full_blocks_count = thrust::transform_reduce(device_begin(_bit_blocks), device_end(_bit_blocks) - 1,
                                                         detail::count_block_bits<block_type>(),
                                                         0,
                                                         thrust::plus<index_t>());

    index_t last_block_count = thrust::transform_reduce(thrust::counting_iterator<index_t>((_number_bit_blocks - 1) * _bits_per_block), thrust::counting_iterator<index_t>(size()),
                                                        detail::count_bits(*this),
                                                        0,
                                                        thrust::plus<index_t>());

    return full_blocks_count + last_block_count;
}


inline bool
bitset::all() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == size();
}


inline bool
bitset::any() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() > 0;
}


inline bool
bitset::none() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == 0;
}

} // namespace stdgpu



#endif // STDGPU_BITSET_DETAIL_H
