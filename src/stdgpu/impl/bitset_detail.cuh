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
#include <stdgpu/limits.h>



namespace stdgpu
{

inline STDGPU_DEVICE_ONLY bool
bitset::set(const index_t n,
            const bool value)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    const sizediv_t positions = sizedivPow2(n, _bits_per_block);
    const index_t block = positions.quot;
    const index_t bit_n = positions.rem;

    block_type set_pattern = static_cast<block_type>(1) << bit_n;
    block_type reset_pattern = numeric_limits<block_type>::max() - set_pattern;

    block_type old;
    stdgpu::atomic_ref<block_type> bit_block(_bit_blocks[block]);
    if (value)
    {
        old = bit_block.fetch_or(set_pattern);
    }
    else
    {
        old = bit_block.fetch_and(reset_pattern);
    }

    return ((old & (static_cast<block_type>(1) << bit_n)) != 0);
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

    const sizediv_t positions = sizedivPow2(n, _bits_per_block);
    const index_t block = positions.quot;
    const index_t bit_n = positions.rem;

    block_type flip_pattern = static_cast<block_type>(1) << bit_n;

    stdgpu::atomic_ref<block_type> bit_block(_bit_blocks[block]);
    block_type old = bit_block.fetch_xor(flip_pattern);

    return ((old & (static_cast<block_type>(1) << bit_n)) != 0);
}


inline STDGPU_DEVICE_ONLY bool
bitset::operator[](const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    const sizediv_t positions = sizedivPow2(n, _bits_per_block);
    const index_t block = positions.quot;
    const index_t bit_n = positions.rem;

    return ((_bit_blocks[block] & (static_cast<block_type>(1) << bit_n)) != 0);
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
