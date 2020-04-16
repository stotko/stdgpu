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

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdgpu/contract.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/utility.h>



namespace stdgpu
{

template <typename T>
deque<T>
deque<T>::createDeviceObject(const index_t& capacity)
{
    STDGPU_EXPECTS(capacity > 0);

    deque<T> result;
    allocator_type a;   // Will be replaced by member
    result._data     = allocator_traits<allocator_type>::allocate(a, capacity);
    result._locks    = mutex_array::createDeviceObject(capacity);
    result._occupied = bitset::createDeviceObject(capacity);
    result._size     = atomic<int>::createDeviceObject();
    result._begin    = atomic<unsigned int>::createDeviceObject();
    result._end      = atomic<unsigned int>::createDeviceObject();
    result._capacity = capacity;

    result._range_indices = vector<index_t>::createDeviceObject(capacity);

    return result;
}

template <typename T>
void
deque<T>::destroyDeviceObject(deque<T>& device_object)
{
    device_object.clear();

    allocator_type a = device_object.get_allocator();   // Will be replaced by member
    allocator_traits<allocator_type>::deallocate(a, device_object._data, device_object._capacity);
    mutex_array::destroyDeviceObject(device_object._locks);
    bitset::destroyDeviceObject(device_object._occupied);
    atomic<int>::destroyDeviceObject(device_object._size);
    atomic<unsigned int>::destroyDeviceObject(device_object._begin);
    atomic<unsigned int>::destroyDeviceObject(device_object._end);
    device_object._capacity = 0;

    vector<index_t>::destroyDeviceObject(device_object._range_indices);
}


template <typename T>
inline STDGPU_HOST_DEVICE typename deque<T>::allocator_type
deque<T>::get_allocator() const
{
    return allocator_type();
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::reference
deque<T>::at(const deque<T>::index_type n)
{
    return const_cast<reference>(static_cast<const deque<T>*>(this)->at(n));
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::const_reference
deque<T>::at(const deque<T>::index_type n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());
    STDGPU_EXPECTS(occupied(n));

    index_t index_to_wrap = static_cast<index_t>(_begin.load()) + n;

    STDGPU_ASSERT(0 <= index_to_wrap);

    return _data[index_to_wrap % _capacity];
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::reference
deque<T>::operator[](const deque<T>::index_type n)
{
    return at(n);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::const_reference
deque<T>::operator[](const deque<T>::index_type n) const
{
    return at(n);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::reference
deque<T>::front()
{
    return const_cast<reference>(static_cast<const deque<T>*>(this)->front());
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::const_reference
deque<T>::front() const
{
    return operator[](0);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::reference
deque<T>::back()
{
    return const_cast<reference>(static_cast<const deque<T>*>(this)->back());
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename deque<T>::const_reference
deque<T>::back() const
{
    return operator[](size() - 1);
}


template <typename T>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
deque<T>::emplace_back(Args&&... args)
{
    return push_back(T(forward<Args>(args)...));
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
deque<T>::push_back(const T& element)
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
    if (current_size < _capacity)
    {
        index_t push_position = static_cast<index_t>(_end.fetch_inc_mod(static_cast<unsigned int>(_capacity)));

        while (!pushed)
        {
            if (_locks[push_position].try_lock())
            {
                // START --- critical section --- START

                if (!occupied(push_position))
                {
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::construct(a, &(_data[push_position]), element);
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


template <typename T>
inline STDGPU_DEVICE_ONLY thrust::pair<T, bool>
deque<T>::pop_back()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    thrust::pair<T, bool> popped = thrust::make_pair(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::deque::pop_back : Object empty\n");
        return popped;
    }

    index_t current_size =  _size--;

    // Check size
    if (current_size > 0)
    {
        index_t pop_position = static_cast<index_t>(_end.fetch_dec_mod(static_cast<unsigned int>(_capacity)));
        pop_position = (pop_position == 0) ? _capacity - 1 : pop_position - 1;  // Manually reconstruct stored value

        while (!popped.second)
        {
            if (_locks[pop_position].try_lock())
            {
                // START --- critical section --- START

                if (occupied(pop_position))
                {
                    bool was_occupied = _occupied.reset(pop_position);
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::construct(a, &popped, _data[pop_position], true);
                    allocator_traits<allocator_type>::destroy(a, &(_data[pop_position]));

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


template <typename T>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
deque<T>::emplace_front(Args&&... args)
{
    return push_front(T(forward<Args>(args)...));
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
deque<T>::push_front(const T& element)
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
    if (current_size < _capacity)
    {
        index_t push_position = static_cast<index_t>(_begin.fetch_dec_mod(static_cast<unsigned int>(_capacity)));
        push_position = (push_position == 0) ? _capacity - 1 : push_position - 1;  // Manually reconstruct stored value

        while (!pushed)
        {
            if (_locks[push_position].try_lock())
            {
                // START --- critical section --- START

                if (!occupied(push_position))
                {
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::construct(a, &(_data[push_position]), element);
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


template <typename T>
inline STDGPU_DEVICE_ONLY thrust::pair<T, bool>
deque<T>::pop_front()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    thrust::pair<T, bool> popped = thrust::make_pair(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::deque::pop_front : Object empty\n");
        return popped;
    }

    index_t current_size =  _size--;

    // Check size
    if (current_size > 0)
    {
        index_t pop_position = static_cast<index_t>(_begin.fetch_inc_mod(static_cast<unsigned int>(_capacity)));

        while (!popped.second)
        {
            if (_locks[pop_position].try_lock())
            {
                // START --- critical section --- START

                if (occupied(pop_position))
                {
                    bool was_occupied = _occupied.reset(pop_position);
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::construct(a, &popped, _data[pop_position], true);
                    allocator_traits<allocator_type>::destroy(a, &(_data[pop_position]));

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


template <typename T>
inline STDGPU_HOST_DEVICE bool
deque<T>::empty() const
{
    return (size() == 0);
}


template <typename T>
inline STDGPU_HOST_DEVICE bool
deque<T>::full() const
{
    return (size() == max_size());
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
deque<T>::size() const
{
    index_t current_size = _size.load();

    // Check boundary cases where the push/pop caused the pointers to be overful/underful
    if (current_size < 0)
    {
        printf("stdgpu::deque::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]. Clamping to 0\n", current_size, _capacity);
        return 0;
    }
    if (current_size > _capacity)
    {
        printf("stdgpu::deque::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]. Clamping to %" STDGPU_PRIINDEX "\n", current_size, _capacity, _capacity);
        return _capacity;
    }

    STDGPU_ENSURES(current_size <= _capacity);
    return current_size;
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
deque<T>::max_size() const
{
    return capacity();
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
deque<T>::capacity() const
{
    return _capacity;
}


template <typename T>
inline void
deque<T>::shrink_to_fit()
{
    // Reject request for performance reasons
}


template <typename T>
inline const T*
deque<T>::data() const
{
    return _data;
}


template <typename T>
inline T*
deque<T>::data()
{
    return _data;
}


template <typename T>
inline void
deque<T>::clear()
{
    if (empty())
    {
        return;
    }

    const index_t begin = static_cast<index_t>(_begin.load());
    const index_t end   = static_cast<index_t>(_end.load());

    // One large block
    if (begin <= end)
    {
        stdgpu::destroy(stdgpu::make_device(_data + begin), stdgpu::make_device(_data + end));
    }
    // Two disconnected blocks
    else
    {
        stdgpu::destroy(stdgpu::device_begin(_data), stdgpu::make_device(_data + end));
        stdgpu::destroy(stdgpu::make_device(_data + begin), stdgpu::device_end(_data));
    }


    _occupied.reset();

    _size.store(0);

    _begin.store(0);
    _end.store(0);

    STDGPU_ENSURES(empty());
    STDGPU_ENSURES(valid());
}


template <typename T>
inline bool
deque<T>::valid() const
{
    // Special case : Zero capacity is valid
    if (capacity() == 0)
    {
        return true;
    }

    return (size_valid()
         && occupied_count_valid()
         && _locks.valid());
}

namespace detail
{

template <typename T>
class deque_collect_positions
{
    public:
        explicit deque_collect_positions(const deque<T>& d)
            : _d(d)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const index_t i)
        {
            if (_d.occupied(i))
            {
                _d._range_indices.push_back(i);
            }
        }

    private:
        deque<T> _d;
};

} // namespace detail


template <typename T>
stdgpu::device_indexed_range<T>
deque<T>::device_range()
{
    _range_indices.clear();

    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                     detail::deque_collect_positions<T>(*this));

    return device_indexed_range<value_type>(_range_indices.device_range(), data());
}


template <typename T>
stdgpu::device_indexed_range<const T>
deque<T>::device_range() const
{
    _range_indices.clear();

    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(size()),
                     detail::deque_collect_positions<T>(*this));

    return device_indexed_range<const value_type>(_range_indices.device_range(), data());
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
deque<T>::occupied(const index_t n) const
{
    return _occupied[n];
}


template <typename T>
bool
deque<T>::occupied_count_valid() const
{
    index_t size_count = size();
    index_t size_sum   = _occupied.count();

    return (size_count == size_sum);
}


template <typename T>
bool
deque<T>::size_valid() const
{
    int current_size = _size.load();
    return (0 <= current_size && current_size <= static_cast<int>(_capacity));
}

} // namespace stdgpu



#endif // STDGPU_DEQUE_DETAIL_H
