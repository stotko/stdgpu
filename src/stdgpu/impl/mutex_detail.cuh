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

#ifndef STDGPU_MUTEX_DETAIL_H
#define STDGPU_MUTEX_DETAIL_H

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <stdgpu/contract.h>



namespace stdgpu
{

inline STDGPU_HOST_DEVICE
mutex_array::reference::reference(const bitset<>::reference& bit_ref)
    : _bit_ref(bit_ref)
{

}


inline STDGPU_DEVICE_ONLY bool
mutex_array::reference::try_lock()
{
    // Change state to LOCKED
    // Test whether it was UNLOCKED previously --> TRUE : This call got the lock, FALSE : Other call got the lock
    return !(_bit_ref = true);
}


inline STDGPU_DEVICE_ONLY void
mutex_array::reference::unlock()
{
    // Change state back to UNLOCKED
    _bit_ref = false;
}


inline STDGPU_DEVICE_ONLY bool
mutex_array::reference::locked() const
{
    return _bit_ref;
}



inline mutex_array
mutex_array::createDeviceObject(const index_t& size)
{
    mutex_array result;
    result._lock_bits = bitset<>::createDeviceObject(size);
    result._size  = size;

    return result;
}


inline void
mutex_array::destroyDeviceObject(mutex_array& device_object)
{
    bitset<>::destroyDeviceObject(device_object._lock_bits);
    device_object._size = 0;
}


inline STDGPU_DEVICE_ONLY mutex_array::reference
mutex_array::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return reference(_lock_bits[n]);
}


inline STDGPU_DEVICE_ONLY const mutex_array::reference
mutex_array::operator[](const index_t n) const
{
    return const_cast<mutex_array*>(this)->operator[](n);
}


inline STDGPU_HOST_DEVICE bool
mutex_array::empty() const
{
    return (size() == 0);
}


inline STDGPU_HOST_DEVICE index_t
mutex_array::size() const
{
    return _size;
}


namespace detail
{

class unlocked
{
    public:
        inline
        explicit unlocked(const mutex_array& lock_bits)
            : _lock_bits(lock_bits)
        {

        }

        inline STDGPU_DEVICE_ONLY bool
        operator()(const index_t i) const
        {
            return !(_lock_bits[i].locked());
        }

    private:
        mutex_array _lock_bits;
};

} // namespace detail


inline bool
mutex_array::valid() const
{
    if (empty())
    {
        return true;
    }

    return thrust::all_of(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                          detail::unlocked(*this));
}



namespace detail
{

template <typename Lockable1>
inline STDGPU_DEVICE_ONLY int
try_lock(int i,
         Lockable1 lock1)
{
    return lock1.try_lock() ? -1 : i;
}

template <typename Lockable1, typename... LockableN>
inline STDGPU_DEVICE_ONLY int
try_lock(int i,
         Lockable1 lock1,
         LockableN... lockn)
{
    if (!lock1.try_lock())
    {
        return i;
    }

    // Success --> lock remaining positions
    int result = try_lock(++i, lockn...);

    // Failure in one of the remaining positions --> unlock own one
    if (result != -1)
    {
        lock1.unlock();
    }

    return result;
}

} // namespace detail


template <typename Lockable1, typename Lockable2, typename... LockableN>
inline STDGPU_DEVICE_ONLY int
try_lock(Lockable1 lock1,
                 Lockable2 lock2,
                 LockableN... lockn)
{
    return detail::try_lock(0, lock1, lock2, lockn...);
}

} // namespace stdgpu



#endif // STDGPU_MUTEX_DETAIL_H
