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

#include <stdgpu/atomic.cuh>
#include <stdgpu/bit.h>
#include <stdgpu/contract.h>



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
                       const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < _bits_per_block);

    return ((bits & (static_cast<block_type>(1) << static_cast<block_type>(n))) != 0);
}


inline STDGPU_DEVICE_ONLY bool
bitset::set(const index_t n,
            const bool value)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return operator[](n) = value;
}


inline STDGPU_DEVICE_ONLY bool
bitset::reset(const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return set(n, false);
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

} // namespace stdgpu



#endif // STDGPU_BITSET_DETAIL_H
