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

#include <stdgpu/algorithm.h>
#include <stdgpu/contract.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/numeric.h>
#include <stdgpu/utility.h>

namespace stdgpu
{

template <typename T, typename Allocator>
vector<T, Allocator>
vector<T, Allocator>::createDeviceObject(const index_t& capacity, const Allocator& allocator)
{
    STDGPU_EXPECTS(capacity > 0);

    vector<T, Allocator> result(
            mutex_array<mutex_default_type, mutex_array_allocator_type>::createDeviceObject(
                    capacity,
                    mutex_array_allocator_type(allocator)),
            bitset<bitset_default_type, bitset_allocator_type>::createDeviceObject(capacity,
                                                                                   bitset_allocator_type(allocator)),
            atomic<int, atomic_allocator_type>::createDeviceObject(atomic_allocator_type(allocator)),
            allocator);
    result._data = detail::createUninitializedDeviceArray<T, allocator_type>(result._allocator, capacity);

    return result;
}

template <typename T, typename Allocator>
void
vector<T, Allocator>::destroyDeviceObject(vector<T, Allocator>& device_object)
{
    if (!detail::is_allocator_destroy_optimizable<value_type, allocator_type>())
    {
        device_object.clear();
    }

    detail::destroyUninitializedDeviceArray<T, allocator_type>(device_object._allocator, device_object._data);
    mutex_array<mutex_default_type, mutex_array_allocator_type>::destroyDeviceObject(device_object._locks);
    bitset<bitset_default_type, bitset_allocator_type>::destroyDeviceObject(device_object._occupied);
    atomic<int, atomic_allocator_type>::destroyDeviceObject(device_object._size);
}

template <typename T, typename Allocator>
inline vector<T, Allocator>::vector(const mutex_array<mutex_default_type, mutex_array_allocator_type>& locks,
                                    const bitset<bitset_default_type, bitset_allocator_type>& occupied,
                                    const atomic<int, atomic_allocator_type>& size,
                                    const Allocator& allocator)
  : _locks(locks)
  , _occupied(occupied)
  , _size(size)
  , _allocator(allocator)
{
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE typename vector<T, Allocator>::allocator_type
vector<T, Allocator>::get_allocator() const noexcept
{
    return _allocator;
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::reference
vector<T, Allocator>::at(const vector<T, Allocator>::index_type n)
{
    return const_cast<vector<T, Allocator>::reference>(static_cast<const vector<T, Allocator>*>(this)->at(n));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::const_reference
vector<T, Allocator>::at(const vector<T, Allocator>::index_type n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < size());
    STDGPU_EXPECTS(occupied(n));

    return operator[](n);
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::reference
vector<T, Allocator>::operator[](const vector<T, Allocator>::index_type n)
{
    return const_cast<vector<T, Allocator>::reference>(static_cast<const vector<T, Allocator>*>(this)->operator[](n));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::const_reference
vector<T, Allocator>::operator[](const vector<T, Allocator>::index_type n) const
{
    return _data[n];
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::reference
vector<T, Allocator>::front()
{
    return const_cast<reference>(static_cast<const vector<T, Allocator>*>(this)->front());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::const_reference
vector<T, Allocator>::front() const
{
    return operator[](0);
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::reference
vector<T, Allocator>::back()
{
    return const_cast<reference>(static_cast<const vector<T, Allocator>*>(this)->back());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY typename vector<T, Allocator>::const_reference
vector<T, Allocator>::back() const
{
    return operator[](size() - 1);
}

template <typename T, typename Allocator>
template <class... Args>
inline STDGPU_DEVICE_ONLY bool
vector<T, Allocator>::emplace_back(Args&&... args)
{
    return push_back(T(forward<Args>(args)...));
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
vector<T, Allocator>::push_back(const T& element)
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
    if (0 <= push_position && push_position < capacity())
    {
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
        printf("stdgpu::vector::push_back : Index out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
               "]\n",
               push_position,
               capacity() - 1);
    }

    return pushed;
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY pair<T, bool>
vector<T, Allocator>::pop_back()
{
    // Value if no element will be popped, i.e. undefined behavior for element of type T
    pair<T, bool> popped(_data[0], false);

    // Preemptive check
    if (empty())
    {
        printf("stdgpu::vector::pop_back : Object empty\n");
        return popped;
    }

    index_t pop_position = --_size;

    // Check position
    if (0 <= pop_position && pop_position < capacity())
    {
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
        printf("stdgpu::vector::pop_back : Index out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
               "]\n",
               pop_position,
               capacity() - 1);
    }

    return popped;
}

namespace detail
{

template <typename T, typename Allocator, typename ValueIterator, bool update_occupancy>
class vector_insert
{
public:
    vector_insert(const vector<T, Allocator>& v, index_t begin, ValueIterator values)
      : _v(v)
      , _begin(begin)
      , _values(values)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        allocator_traits<typename vector<T, Allocator>::allocator_type>::construct(_v._allocator,
                                                                                   &(_v._data[_begin + i]),
                                                                                   _values[i]);

        if (update_occupancy)
        {
            _v._occupied.set(_begin + i);
        }
    }

private:
    vector<T, Allocator> _v;
    index_t _begin;
    ValueIterator _values;
};

template <typename T, typename Allocator, bool update_occupancy>
class vector_erase
{
public:
    vector_erase(const vector<T, Allocator>& v, const index_t begin)
      : _v(v)
      , _begin(begin)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        allocator_traits<typename vector<T, Allocator>::allocator_type>::destroy(_v._allocator,
                                                                                 &(_v._data[_begin + i]));

        if (update_occupancy)
        {
            _v._occupied.reset(_begin + i);
        }
    }

private:
    vector<T, Allocator> _v;
    index_t _begin;
};

template <typename T, typename Allocator>
void
vector_clear_iota(vector<T, Allocator>& v, const T& value)
{
    iota(execution::device, device_begin(v.data()), device_end(v.data()), value);
    v._occupied.set();
    v._size.store(v.capacity());
}

} // namespace detail

template <typename T, typename Allocator>
template <typename ValueIterator, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_iterator_v<ValueIterator>)>
inline void
vector<T, Allocator>::insert(device_ptr<const T> position, ValueIterator begin, ValueIterator end)
{
    if (position != device_end())
    {
        printf("stdgpu::vector::insert : Position not equal to device_end()\n");
        return;
    }

    index_t N = static_cast<index_t>(end - begin);
    index_t new_size = size() + N;

    if (new_size > capacity())
    {
        printf("stdgpu::vector::insert : Unable to insert all values: New size %" STDGPU_PRIINDEX
               " would exceed capacity %" STDGPU_PRIINDEX "\n",
               new_size,
               capacity());
        return;
    }

    for_each_index(execution::device,
                   N,
                   detail::vector_insert<T, Allocator, ValueIterator, true>(*this, size(), begin));

    _size.store(new_size);
}

template <typename T, typename Allocator>
inline void
vector<T, Allocator>::erase(device_ptr<const T> begin, device_ptr<const T> end)
{
    if (end != device_end())
    {
        printf("stdgpu::vector::erase : End iterator not equal to device_end()\n");
        return;
    }

    index_t N = static_cast<index_t>(end - begin);
    index_t new_size = size() - N;

    if (new_size < 0)
    {
        printf("stdgpu::vector::erase : Unable to erase all values: New size %" STDGPU_PRIINDEX " would be invalid\n",
               new_size);
        return;
    }

    for_each_index(execution::device, N, detail::vector_erase<T, Allocator, true>(*this, new_size));

    _size.store(new_size);
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
vector<T, Allocator>::empty() const
{
    return (size() == 0);
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
vector<T, Allocator>::full() const
{
    return (size() == max_size());
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
vector<T, Allocator>::size() const
{
    index_t current_size = _size.load();

    // Check boundary cases where the push/pop caused the pointers to be overful/underful
    if (current_size < 0)
    {
        printf("stdgpu::vector::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
               "]. Clamping to 0\n",
               current_size,
               capacity());
        return 0;
    }
    if (current_size > capacity())
    {
        printf("stdgpu::vector::size : Size out of bounds: %" STDGPU_PRIINDEX " not in [0, %" STDGPU_PRIINDEX
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
vector<T, Allocator>::max_size() const noexcept
{
    return capacity();
}

template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
vector<T, Allocator>::capacity() const noexcept
{
    return _occupied.size();
}

template <typename T, typename Allocator>
inline void
vector<T, Allocator>::shrink_to_fit()
{
    // Reject request for performance reasons
}

template <typename T, typename Allocator>
inline const T*
vector<T, Allocator>::data() const noexcept
{
    return _data;
}

template <typename T, typename Allocator>
inline T*
vector<T, Allocator>::data() noexcept
{
    return _data;
}

template <typename T, typename Allocator>
inline void
vector<T, Allocator>::clear()
{
    if (empty())
    {
        return;
    }

    if (!detail::is_allocator_destroy_optimizable<value_type, allocator_type>())
    {
        const index_t current_size = size();

        detail::unoptimized_destroy(execution::device,
                                    stdgpu::device_begin(_data),
                                    stdgpu::device_begin(_data) + current_size);
    }

    _occupied.reset();

    _size.store(0);

    STDGPU_ENSURES(empty());
    STDGPU_ENSURES(valid());
}

template <typename T, typename Allocator>
inline bool
vector<T, Allocator>::valid() const
{
    // Special case : Zero capacity is valid
    if (capacity() == 0)
    {
        return true;
    }

    return (size_valid() && occupied_count_valid() && _locks.valid());
}

template <typename T, typename Allocator>
device_ptr<T>
vector<T, Allocator>::device_begin()
{
    return stdgpu::device_begin(_data);
}

template <typename T, typename Allocator>
device_ptr<T>
vector<T, Allocator>::device_end()
{
    return device_begin() + size();
}

template <typename T, typename Allocator>
device_ptr<const T>
vector<T, Allocator>::device_begin() const
{
    return stdgpu::device_begin(_data);
}

template <typename T, typename Allocator>
device_ptr<const T>
vector<T, Allocator>::device_end() const
{
    return device_begin() + size();
}

template <typename T, typename Allocator>
device_ptr<const T>
vector<T, Allocator>::device_cbegin() const
{
    return stdgpu::device_cbegin(_data);
}

template <typename T, typename Allocator>
device_ptr<const T>
vector<T, Allocator>::device_cend() const
{
    return device_cbegin() + size();
}

template <typename T, typename Allocator>
stdgpu::device_range<T>
vector<T, Allocator>::device_range()
{
    return stdgpu::device_range<T>(_data, size());
}

template <typename T, typename Allocator>
stdgpu::device_range<const T>
vector<T, Allocator>::device_range() const
{
    return stdgpu::device_range<const T>(_data, size());
}

template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
vector<T, Allocator>::occupied(const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < capacity());

    return _occupied[n];
}

template <typename T, typename Allocator>
bool
vector<T, Allocator>::occupied_count_valid() const
{
    index_t size_count = size();
    index_t size_sum = _occupied.count();

    return (size_count == size_sum);
}

template <typename T, typename Allocator>
bool
vector<T, Allocator>::size_valid() const
{
    int current_size = _size.load();
    return (0 <= current_size && current_size <= static_cast<int>(capacity()));
}

} // namespace stdgpu

#endif // STDGPU_VECTOR_DETAIL_H
