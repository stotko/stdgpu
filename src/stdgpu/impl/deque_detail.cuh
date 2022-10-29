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

#ifndef STDGPU_DEQUE_DETAIL_H
#define STDGPU_DEQUE_DETAIL_H

#include <stdgpu/contract.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/numeric.h>
#include <stdgpu/utility.h>

namespace stdgpu
{

template <typename T, typename Allocator>
deque<T, Allocator>
deque<T, Allocator>::createDeviceObject(const index_t& capacity, const Allocator& allocator)
{
    STDGPU_EXPECTS(capacity > 0);

    deque<T, Allocator> result(
            mutex_array<mutex_default_type, mutex_array_allocator_type>::createDeviceObject(
                    capacity,
                    mutex_array_allocator_type(allocator)),
            bitset<bitset_default_type, bitset_allocator_type>::createDeviceObject(capacity,
                                                                                   bitset_allocator_type(allocator)),
            atomic<int, atomic_int_allocator_type>::createDeviceObject(atomic_int_allocator_type(allocator)),
            atomic<unsigned int, atomic_uint_allocator_type>::createDeviceObject(atomic_uint_allocator_type(allocator)),
            atomic<unsigned int, atomic_uint_allocator_type>::createDeviceObject(atomic_uint_allocator_type(allocator)),
            allocator);
    result._data = detail::createUninitializedDeviceArray<T, allocator_type>(result._allocator, capacity);
    result._range_indices =
            detail::createUninitializedDeviceArray<index_t, index_allocator_type>(result._index_allocator, capacity);

    return result;
}

template <typename T, typename Allocator>
void
deque<T, Allocator>::destroyDeviceObject(deque<T, Allocator>& device_object)
{
    if (!detail::is_allocator_destroy_optimizable<value_type, allocator_type>())
    {
        device_object.clear();
    }

    detail::destroyUninitializedDeviceArray<T, allocator_type>(device_object._allocator, device_object._data);
    detail::destroyUninitializedDeviceArray<index_t, index_allocator_type>(device_object._index_allocator,
                                                                           device_object._range_indices);
    mutex_array<mutex_default_type, mutex_array_allocator_type>::destroyDeviceObject(device_object._locks);
    bitset<bitset_default_type, bitset_allocator_type>::destroyDeviceObject(device_object._occupied);
    atomic<int, atomic_int_allocator_type>::destroyDeviceObject(device_object._size);
    atomic<unsigned int, atomic_uint_allocator_type>::destroyDeviceObject(device_object._begin);
    atomic<unsigned int, atomic_uint_allocator_type>::destroyDeviceObject(device_object._end);
}

template <typename T, typename Allocator>
inline deque<T, Allocator>::deque(const mutex_array<mutex_default_type, mutex_array_allocator_type>& locks,
                                  const bitset<bitset_default_type, bitset_allocator_type>& occupied,
                                  const atomic<int, atomic_int_allocator_type>& size,
                                  const atomic<unsigned int, atomic_uint_allocator_type>& begin,
                                  const atomic<unsigned int, atomic_uint_allocator_type>& end,
                                  const Allocator& allocator)
  : _locks(locks)
  , _occupied(occupied)
  , _size(size)
  , _begin(begin)
  , _end(end)
  , _allocator(allocator)
  , _index_allocator(allocator)
{
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE typename deque<T, Allocator>::allocator_type
deque<T, Allocator>::get_allocator() const noexcept
{
    return _allocator;
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::reference
deque<T, Allocator>::at(const deque<T, Allocator>::index_type n)
{
    return const_cast<reference>(static_cast<const deque<T, Allocator>*>(this)->at(n));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::const_reference
deque<T, Allocator>::at(const deque<T, Allocator>::index_type n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());
    STDGPU_EXPECTS(occupied(n));

    return operator[](n);
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::reference
deque<T, Allocator>::operator[](const deque<T, Allocator>::index_type n)
{
    return const_cast<reference>(static_cast<const deque<T, Allocator>*>(this)->operator[](n));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::const_reference
deque<T, Allocator>::operator[](const deque<T, Allocator>::index_type n) const
{
    index_t index_to_wrap = static_cast<index_t>(_begin.load()) + n;
    return _data[index_to_wrap % capacity()];
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::reference
deque<T, Allocator>::front()
{
    return const_cast<reference>(static_cast<const deque<T, Allocator>*>(this)->front());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::const_reference
deque<T, Allocator>::front() const
{
    return operator[](0);
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::reference
deque<T, Allocator>::back()
{
    return const_cast<reference>(static_cast<const deque<T, Allocator>*>(this)->back());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename deque<T, Allocator>::const_reference
deque<T, Allocator>::back() const
{
    return operator[](size() - 1);
}

template <typename T, typename Allocator>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
deque<T, Allocator>::emplace_back(Args&&... args)
{
    return push_back(T(forward<Args>(args)...));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
deque<T, Allocator>::push_back(const T& element)
{
    bool pushed = false;

    // Preemptive check
    if (full())
    {
        printf("stdgpu::deque::push_back : Object full\n");
        return pushed;
    }

    index_t current_size = _size++;

    // Check size
    if (current_size < capacity())
    {
        index_t push_position = static_cast<index_t>(_end.fetch_inc_mod(static_cast<unsigned int>(capacity())));

        while (!pushed)
        {
            if (_locks[push_position].try_lock())
            {
                // START --- critical section --- START

                if (!occupied(push_position))
                {
                    allocator_traits<allocator_type>::construct(_allocator, &(_data[push_position]), element);
                    bool was_occupied = _occupied.set(push_position);
                    pushed = true;

                    if (was_occupied)
                    {
                        printf("stdgpu::deque::push_back : Expected entry to be not occupied but actually was\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[push_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::deque::push_back : Unable to push element to full queue\n");
    }

    return pushed;
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY pair<T, bool>
deque<T, Allocator>::pop_back()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    pair<T, bool> popped(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::deque::pop_back : Object empty\n");
        return popped;
    }

    index_t current_size = _size--;

    // Check size
    if (current_size > 0)
    {
        index_t pop_position = static_cast<index_t>(_end.fetch_dec_mod(static_cast<unsigned int>(capacity())));
        pop_position = (pop_position == 0) ? capacity() - 1 : pop_position - 1; // Manually reconstruct stored value

        while (!popped.second)
        {
            if (_locks[pop_position].try_lock())
            {
                // START --- critical section --- START

                if (occupied(pop_position))
                {
                    bool was_occupied = _occupied.reset(pop_position);
                    allocator_traits<allocator_type>::construct(_allocator, &popped, _data[pop_position], true);
                    allocator_traits<allocator_type>::destroy(_allocator, &(_data[pop_position]));

                    if (!was_occupied)
                    {
                        printf("stdgpu::deque::pop_back : Expected entry to be occupied but actually was not\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[pop_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::deque::pop_back : Unable to pop element from empty queue\n");
    }

    return popped;
}

template <typename T, typename Allocator>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
deque<T, Allocator>::emplace_front(Args&&... args)
{
    return push_front(T(forward<Args>(args)...));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
deque<T, Allocator>::push_front(const T& element)
{
    bool pushed = false;

    // Preemptive check
    if (full())
    {
        printf("stdgpu::deque::push_front : Object full\n");
        return pushed;
    }

    index_t current_size = _size++;

    // Check size
    if (current_size < capacity())
    {
        index_t push_position = static_cast<index_t>(_begin.fetch_dec_mod(static_cast<unsigned int>(capacity())));
        push_position = (push_position == 0) ? capacity() - 1 : push_position - 1; // Manually reconstruct stored value

        while (!pushed)
        {
            if (_locks[push_position].try_lock())
            {
                // START --- critical section --- START

                if (!occupied(push_position))
                {
                    allocator_traits<allocator_type>::construct(_allocator, &(_data[push_position]), element);
                    bool was_occupied = _occupied.set(push_position);
                    pushed = true;

                    if (was_occupied)
                    {
                        printf("stdgpu::deque::push_front : Expected entry to be not occupied but actually was\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[push_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::deque::push_front : Unable to push element to full queue\n");
    }

    return pushed;
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY pair<T, bool>
deque<T, Allocator>::pop_front()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    pair<T, bool> popped(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::deque::pop_front : Object empty\n");
        return popped;
    }

    index_t current_size = _size--;

    // Check size
    if (current_size > 0)
    {
        index_t pop_position = static_cast<index_t>(_begin.fetch_inc_mod(static_cast<unsigned int>(capacity())));

        while (!popped.second)
        {
            if (_locks[pop_position].try_lock())
            {
                // START --- critical section --- START

                if (occupied(pop_position))
                {
                    bool was_occupied = _occupied.reset(pop_position);
                    allocator_traits<allocator_type>::construct(_allocator, &popped, _data[pop_position], true);
                    allocator_traits<allocator_type>::destroy(_allocator, &(_data[pop_position]));

                    if (!was_occupied)
                    {
                        printf("stdgpu::deque::pop_front : Expected entry to be occupied but actually was not\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[pop_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::deque::pop_front : Unable to pop element from empty queue\n");
    }

    return popped;
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
deque<T, Allocator>::empty() const
{
    return (size() == 0);
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
deque<T, Allocator>::full() const
{
    return (size() == max_size());
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
deque<T, Allocator>::size() const
{
    index_t current_size = _size.load();

    // Check boundary cases where the push/pop caused the pointers to be overful/underful
    if (current_size < 0)
    {
        printf("stdgpu::deque::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
               "]. Clamping to 0\n",
               current_size,
               capacity());
        return 0;
    }
    if (current_size > capacity())
    {
        printf("stdgpu::deque::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
               "]. Clamping to %" STDGPU_PRIINDEX "\n",
               current_size,
               capacity(),
               capacity());
        return capacity();
    }

    STDGPU_ENSURES(current_size <= capacity());
    return current_size;
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
deque<T, Allocator>::max_size() const noexcept
{
    return capacity();
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
deque<T, Allocator>::capacity() const noexcept
{
    return _occupied.size();
}

template <typename T, typename Allocator>
inline void
deque<T, Allocator>::shrink_to_fit()
{
    // Reject request for performance reasons
}

template <typename T, typename Allocator>
inline const T*
deque<T, Allocator>::data() const noexcept
{
    return _data;
}

template <typename T, typename Allocator>
inline T*
deque<T, Allocator>::data() noexcept
{
    return _data;
}

template <typename T, typename Allocator>
inline void
deque<T, Allocator>::clear()
{
    if (empty())
    {
        return;
    }

    if (!detail::is_allocator_destroy_optimizable<value_type, allocator_type>())
    {
        const index_t begin = static_cast<index_t>(_begin.load());
        const index_t end = static_cast<index_t>(_end.load());

        // Full, i.e. one large block and begin == end
        if (full())
        {
            detail::unoptimized_destroy(execution::device, device_begin(_data), device_end(_data));
        }
        // One large block
        else if (begin <= end)
        {
            detail::unoptimized_destroy(execution::device, make_device(_data + begin), make_device(_data + end));
        }
        // Two disconnected blocks
        else
        {
            detail::unoptimized_destroy(execution::device, device_begin(_data), make_device(_data + end));
            detail::unoptimized_destroy(execution::device, make_device(_data + begin), device_end(_data));
        }
    }

    _occupied.reset();

    _size.store(0);

    _begin.store(0);
    _end.store(0);

    STDGPU_ENSURES(empty());
    STDGPU_ENSURES(valid());
}

template <typename T, typename Allocator>
inline bool
deque<T, Allocator>::valid() const
{
    // Special case : Zero capacity is valid
    if (capacity() == 0)
    {
        return true;
    }

    return (size_valid() && occupied_count_valid() && _locks.valid());
}

template <typename T, typename Allocator>
stdgpu::device_indexed_range<T>
deque<T, Allocator>::device_range()
{
    const index_t begin = static_cast<index_t>(_begin.load());
    const index_t end = static_cast<index_t>(_end.load());

    // Full, i.e. one large block and begin == end
    if (full())
    {
        iota(execution::device, device_begin(_range_indices), device_end(_range_indices), 0);
    }
    // One large block, including empty block
    else if (begin <= end)
    {
        iota(execution::device, device_begin(_range_indices), device_begin(_range_indices) + (end - begin), begin);
    }
    // Two disconnected blocks
    else
    {
        iota(execution::device, device_begin(_range_indices), device_begin(_range_indices) + end, 0);
        iota(execution::device,
             device_begin(_range_indices) + end,
             device_begin(_range_indices) + (end + capacity() - begin),
             begin);
    }

    return device_indexed_range<value_type>(stdgpu::device_range<index_t>(_range_indices, size()), data());
}

template <typename T, typename Allocator>
stdgpu::device_indexed_range<const T>
deque<T, Allocator>::device_range() const
{
    const index_t begin = static_cast<index_t>(_begin.load());
    const index_t end = static_cast<index_t>(_end.load());

    // Full, i.e. one large block and begin == end
    if (full())
    {
        iota(execution::device, device_begin(_range_indices), device_end(_range_indices), 0);
    }
    // One large block, including empty block
    else if (begin <= end)
    {
        iota(execution::device, device_begin(_range_indices), device_begin(_range_indices) + (end - begin), begin);
    }
    // Two disconnected blocks
    else
    {
        iota(execution::device, device_begin(_range_indices), device_begin(_range_indices) + end, 0);
        iota(execution::device,
             device_begin(_range_indices) + end,
             device_begin(_range_indices) + (end + capacity() - begin),
             begin);
    }

    return device_indexed_range<const value_type>(stdgpu::device_range<index_t>(_range_indices, size()), data());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
deque<T, Allocator>::occupied(const index_t n) const
{
    return _occupied[n];
}

template <typename T, typename Allocator>
bool
deque<T, Allocator>::occupied_count_valid() const
{
    index_t size_count = size();
    index_t size_sum = _occupied.count();

    return (size_count == size_sum);
}

template <typename T, typename Allocator>
bool
deque<T, Allocator>::size_valid() const
{
    int current_size = _size.load();
    return (0 <= current_size && current_size <= static_cast<int>(capacity()));
}

} // namespace stdgpu

#endif // STDGPU_DEQUE_DETAIL_H
