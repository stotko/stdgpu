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

#include <stdgpu/contract.h>

namespace stdgpu
{

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE
mutex_array<Block, Allocator>::reference::reference(const typename bitset<Block, Allocator>::reference& bit_ref)
  : _bit_ref(bit_ref)
{
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
mutex_array<Block, Allocator>::reference::try_lock()
{
    // Change state to LOCKED
    // Test whether it was UNLOCKED previously --> TRUE : This call got the lock, FALSE : Other call got the lock
    return !(_bit_ref = true);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY void
mutex_array<Block, Allocator>::reference::unlock()
{
    // Change state back to UNLOCKED
    _bit_ref = false;
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
mutex_array<Block, Allocator>::reference::locked() const
{
    return _bit_ref;
}

template <typename Block, typename Allocator>
inline mutex_array<Block, Allocator>
mutex_array<Block, Allocator>::createDeviceObject(const index_t& size, const Allocator& allocator)
{
    mutex_array<Block, Allocator> result(bitset<Block, Allocator>::createDeviceObject(size, allocator));

    return result;
}

template <typename Block, typename Allocator>
inline void
mutex_array<Block, Allocator>::destroyDeviceObject(mutex_array<Block, Allocator>& device_object)
{
    bitset<Block, Allocator>::destroyDeviceObject(device_object._lock_bits);
}

template <typename Block, typename Allocator>
inline mutex_array<Block, Allocator>::mutex_array(const bitset<Block, Allocator>& lock_bits)
  : _lock_bits(lock_bits)
{
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE typename mutex_array<Block, Allocator>::allocator_type
mutex_array<Block, Allocator>::get_allocator() const noexcept
{
    return _lock_bits.get_allocator();
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY typename mutex_array<Block, Allocator>::reference
mutex_array<Block, Allocator>::operator[](const index_t n)
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());

    return reference(_lock_bits[n]);
}

template <typename Block, typename Allocator>
inline STDGPU_DEVICE_ONLY const typename mutex_array<Block, Allocator>::reference
mutex_array<Block, Allocator>::operator[](const index_t n) const
{
    return const_cast<mutex_array<Block, Allocator>*>(this)->operator[](n);
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE bool
mutex_array<Block, Allocator>::empty() const noexcept
{
    return (size() == 0);
}

template <typename Block, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
mutex_array<Block, Allocator>::size() const noexcept
{
    return _lock_bits.size();
}

template <typename Block, typename Allocator>
inline bool
mutex_array<Block, Allocator>::valid() const
{
    return _lock_bits.count() == 0;
}

namespace detail
{

template <typename Lockable1>
inline STDGPU_DEVICE_ONLY int
try_lock(int i, Lockable1 lock1)
{
    return lock1.try_lock() ? -1 : i;
}

template <typename Lockable1, typename... LockableN>
inline STDGPU_DEVICE_ONLY int
try_lock(int i, Lockable1 lock1, LockableN... lockn)
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
try_lock(Lockable1 lock1, Lockable2 lock2, LockableN... lockn)
{
    return detail::try_lock(0, lock1, lock2, lockn...);
}

} // namespace stdgpu

#endif // STDGPU_MUTEX_DETAIL_H
