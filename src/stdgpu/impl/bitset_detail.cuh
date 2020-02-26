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
#include <stdgpu/contract.h>
#include <stdgpu/cstdlib.h>



namespace stdgpu
{

inline STDGPU_HOST_DEVICE
bitset::reference::reference(bitset::reference::block_type* bit_block,
                             const index_t bit_n)
    : _bit_block(bit_block),
      _bit_n(bit_n)
{

}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::operator=(bool x)
{
    block_type set_pattern = static_cast<block_type>(1) << _bit_n;
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


inline STDGPU_DEVICE_ONLY bool
bitset::reference::operator=(const reference& x)
{
    return operator=(static_cast<bool>(x));
}


inline STDGPU_DEVICE_ONLY
bitset::reference::operator bool() const
{
    return bit(*_bit_block, _bit_n);
}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::operator~() const
{
    return !operator bool();
}


inline STDGPU_DEVICE_ONLY bool
bitset::reference::flip()
{
    block_type flip_pattern = static_cast<block_type>(1) << _bit_n;

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

    return ((bits & (static_cast<block_type>(1) << n)) != 0);
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

    const sizediv_t positions = sizedivPow2(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block));

    return reference(_bit_blocks + positions.quot, static_cast<index_t>(positions.rem));
}


inline STDGPU_DEVICE_ONLY bitset::reference
bitset::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    const sizediv_t positions = sizedivPow2(static_cast<std::size_t>(n), static_cast<std::size_t>(_bits_per_block));

    return reference(_bit_blocks + positions.quot, static_cast<index_t>(positions.rem));
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
