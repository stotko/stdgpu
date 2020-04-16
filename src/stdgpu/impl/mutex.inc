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

#include <stdgpu/mutex.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>



namespace stdgpu
{

mutex_array
mutex_array::createDeviceObject(const index_t& size)
{
    mutex_array result;
    result._lock_bits = bitset::createDeviceObject(size);
    result._size  = size;

    return result;
}


void
mutex_array::destroyDeviceObject(mutex_array& device_object)
{
    bitset::destroyDeviceObject(device_object._lock_bits);
    device_object._size = 0;
}


namespace detail
{

class unlocked
{
    public:
        explicit unlocked(const mutex_array& lock_bits)
            : _lock_bits(lock_bits)
        {

        }

        STDGPU_DEVICE_ONLY bool
        operator()(const index_t i) const
        {
            return !(_lock_bits[i].locked());
        }

    private:
        mutex_array _lock_bits;
};

} // namespace detail


bool
mutex_array::valid() const
{
    if (empty())
    {
        return true;
    }

    return thrust::all_of(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                          detail::unlocked(*this));
}

} // namespace stdgpu


