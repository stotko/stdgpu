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

#ifndef STDGPU_VECTOR_DETAIL_H
#define STDGPU_VECTOR_DETAIL_H

#include <stdgpu/contract.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/utility.h>



namespace stdgpu
{

template <typename T>
vector<T>
vector<T>::createDeviceObject(const index_t& capacity)
{
    STDGPU_EXPECTS(capacity > 0);

    vector<T> result;
    allocator_type a;   // Will be replaced by member
    result._data     = allocator_traits<allocator_type>::allocate(a, capacity);
    result._locks    = mutex_array::createDeviceObject(capacity);
    result._occupied = bitset::createDeviceObject(capacity);
    result._size     = atomic<int>::createDeviceObject();
    result._capacity = capacity;

    return result;
}

template <typename T>
void
vector<T>::destroyDeviceObject(vector<T>& device_object)
{
    device_object.clear();

    allocator_type a = device_object.get_allocator();   // Will be replaced by member
    allocator_traits<allocator_type>::deallocate(a, device_object._data, device_object._capacity);
    mutex_array::destroyDeviceObject(device_object._locks);
    bitset::destroyDeviceObject(device_object._occupied);
    atomic<int>::destroyDeviceObject(device_object._size);
    device_object._capacity = 0;
}


template <typename T>
inline STDGPU_HOST_DEVICE typename vector<T>::allocator_type
vector<T>::get_allocator() const
{
    return allocator_type();
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::reference
vector<T>::at(const vector<T>::index_type n)
{
    return const_cast<vector<T>::reference>(static_cast<const vector<T>*>(this)->at(n));
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::const_reference
vector<T>::at(const vector<T>::index_type n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());
    STDGPU_EXPECTS(occupied(n));

    return _data[n];
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::reference
vector<T>::operator[](const vector<T>::index_type n)
{
    return at(n);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::const_reference
vector<T>::operator[](const vector<T>::index_type n) const
{
    return at(n);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::reference
vector<T>::front()
{
    return const_cast<reference>(static_cast<const vector<T>*>(this)->front());
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::const_reference
vector<T>::front() const
{
    return operator[](0);
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::reference
vector<T>::back()
{
    return const_cast<reference>(static_cast<const vector<T>*>(this)->back());
}


template <typename T>
inline STDGPU_DEVICE_ONLY typename vector<T>::const_reference
vector<T>::back() const
{
    return operator[](size() - 1);
}


template <typename T>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
vector<T>::emplace_back(Args&&... args)
{
    return push_back(T(forward<Args>(args)...));
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
vector<T>::push_back(const T& element)
{
    bool pushed = false;

    // Preemptive check
    if (full())
    {
        printf("stdgpu::vector::push_back : Object full\n");
        return pushed;
    }

    index_t push_position = _size++;

    // Check position
    if (0 <= push_position && push_position < _capacity)
    {
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
                        printf("stdgpu::vector::push_back : Expected entry to be not occupied but actually was\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[push_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::vector::push_back : Index out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]\n", push_position, _capacity - 1);
    }

    return pushed;
}


template <typename T>
inline STDGPU_DEVICE_ONLY thrust::pair<T, bool>
vector<T>::pop_back()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    thrust::pair<T, bool> popped = thrust::make_pair(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::vector::pop_back : Object empty\n");
        return popped;
    }

    index_t pop_position = --_size;

    // Check position
    if (0 <= pop_position && pop_position < _capacity)
    {
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
                        printf("stdgpu::vector::pop_back : Expected entry to be occupied but actually was not\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[pop_position].unlock();
            }
        }
    }
    else
    {
        printf("stdgpu::vector::pop_back : Index out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]\n", pop_position, _capacity - 1);
    }

    return popped;
}


template <typename T>
inline STDGPU_HOST_DEVICE bool
vector<T>::empty() const
{
    return (size() == 0);
}


template <typename T>
inline STDGPU_HOST_DEVICE bool
vector<T>::full() const
{
    return (size() == max_size());
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
vector<T>::size() const
{
    index_t current_size = _size.load();

    // Check boundary cases where the push/pop caused the pointers to be overful/underful
    if (current_size < 0)
    {
        printf("stdgpu::vector::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]. Clamping to 0\n", current_size, _capacity);
        return 0;
    }
    if (current_size > _capacity)
    {
        printf("stdgpu::vector::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX "]. Clamping to %" STDGPU_PRIINDEX "\n", current_size, _capacity, _capacity);
        return _capacity;
    }

    STDGPU_ENSURES(current_size <= _capacity);
    return current_size;
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
vector<T>::max_size() const
{
    return capacity();
}


template <typename T>
inline STDGPU_HOST_DEVICE index_t
vector<T>::capacity() const
{
    return _capacity;
}


template <typename T>
inline void
vector<T>::shrink_to_fit()
{
    // Reject request for performance reasons
}


template <typename T>
inline const T*
vector<T>::data() const
{
    return _data;
}


template <typename T>
inline T*
vector<T>::data()
{
    return _data;
}


template <typename T>
inline void
vector<T>::clear()
{
    if (empty())
    {
        return;
    }

    const index_t current_size = size();

    stdgpu::destroy(stdgpu::device_begin(_data), stdgpu::device_begin(_data) + current_size);

    _occupied.reset();

    _size.store(0);

    STDGPU_ENSURES(empty());
    STDGPU_ENSURES(valid());
}


template <typename T>
inline bool
vector<T>::valid() const
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


template <typename T>
device_ptr<T>
vector<T>::device_begin()
{
    return stdgpu::device_begin(_data);
}


template <typename T>
device_ptr<T>
vector<T>::device_end()
{
    return device_begin() + size();
}


template <typename T>
device_ptr<const T>
vector<T>::device_begin() const
{
    return stdgpu::device_begin(_data);
}


template <typename T>
device_ptr<const T>
vector<T>::device_end() const
{
    return device_begin() + size();
}


template <typename T>
device_ptr<const T>
vector<T>::device_cbegin() const
{
    return stdgpu::device_cbegin(_data);
}


template <typename T>
device_ptr<const T>
vector<T>::device_cend() const
{
    return device_cbegin() + size();
}


template <typename T>
stdgpu::device_range<T>
vector<T>::device_range()
{
    return stdgpu::device_range<T>(_data, size());
}


template <typename T>
stdgpu::device_range<const T>
vector<T>::device_range() const
{
    return stdgpu::device_range<const T>(_data, size());
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
vector<T>::occupied(const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < capacity());

    return _occupied[n];
}


template <typename T>
bool
vector<T>::occupied_count_valid() const
{
    index_t size_count = size();
    index_t size_sum   = _occupied.count();

    return (size_count == size_sum);
}


template <typename T>
bool
vector<T>::size_valid() const
{
    int current_size = _size.load();
    return (0 <= current_size && current_size <= static_cast<int>(_capacity));
}

} // namespace stdgpu



#endif // STDGPU_VECTOR_DETAIL_H
