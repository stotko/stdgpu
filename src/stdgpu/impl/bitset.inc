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

#include <stdgpu/bitset.cuh>

#include <limits>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <stdgpu/bit.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



namespace stdgpu
{

namespace detail
{

index_t
div_up(const index_t a,
       const index_t b)
{
    STDGPU_EXPECTS(a >= 0);
    STDGPU_EXPECTS(b > 0);

    index_t result = (a % b != 0) ? (a / b + 1) : (a / b);

    STDGPU_ENSURES(result * b >= a);

    return result;
}

class set_bits
{
    public:
        set_bits(const bitset& bits,
                 const bool value)
            : _bits(bits),
              _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const index_t i)
        {
            _bits.set(i, _value);
        }

    private:
        bitset _bits;
        const bool _value;
};

class flip_bits
{
    public:
        explicit flip_bits(const bitset& bits)
            : _bits(bits)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const index_t i)
        {
            _bits.flip(i);
        }

    private:
        bitset _bits;
};

template <typename T>
struct count_bits
{
    STDGPU_HOST_DEVICE int
    operator()(const T pattern) const
    {
        return popcount(pattern);
    }
};

} // namespace detail


bitset
bitset::createDeviceObject(const index_t& size)
{
    bitset result;
    result._bits_per_block      = std::numeric_limits<block_type>::digits;
    result._number_bit_blocks   = detail::div_up(size, result._bits_per_block);
    result._bit_blocks          = createDeviceArray<block_type>(result._number_bit_blocks, static_cast<block_type>(0));
    result._size                = size;

    return result;
}


void
bitset::destroyDeviceObject(bitset& device_object)
{
    destroyDeviceArray<block_type>(device_object._bit_blocks);
    device_object._bit_blocks   = nullptr;
    device_object._size         = 0;
}


void
bitset::set()
{
    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                     detail::set_bits(*this, true));

    STDGPU_ENSURES(count() == size());
}


void
bitset::reset()
{
    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                     detail::set_bits(*this, false));

    STDGPU_ENSURES(count() == 0);
}


void
bitset::flip()
{
    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                     detail::flip_bits(*this));
}


index_t
bitset::count() const
{
    if (size() == 0)
    {
        return 0;
    }

    return static_cast<index_t>(thrust::transform_reduce(device_begin(_bit_blocks), device_end(_bit_blocks),
                                                         detail::count_bits<block_type>(),
                                                         block_type(0),
                                                         thrust::plus<block_type>()));
}


bool
bitset::all() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == size();
}


bool
bitset::any() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() > 0;
}


bool
bitset::none() const
{
    if (size() == 0)
    {
        return false;
    }

    return count() == 0;
}

} // namespace stdgpu


