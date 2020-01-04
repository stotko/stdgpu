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

#ifndef STDGPU_UNORDERED_BASE_DETAIL_H
#define STDGPU_UNORDERED_BASE_DETAIL_H

#include <algorithm>
#include <cmath>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <stdgpu/bit.h>
#include <stdgpu/config.h>
#include <stdgpu/contract.h>
#include <stdgpu/cstdlib.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/utility.h>



namespace stdgpu
{

namespace detail
{

inline index_t
next_pow2(const index_t capacity)
{
    STDGPU_EXPECTS(capacity > 0);

    index_t result = static_cast<index_t>(1) << static_cast<index_t>(std::ceil(std::log2(capacity)));

    STDGPU_ENSURES(result >= capacity);
    STDGPU_ENSURES(ispow2<std::size_t>(result));

    return result;
}


inline index_t
expected_collisions(const index_t bucket_count,
                    const index_t capacity)
{
    STDGPU_EXPECTS(bucket_count > 0);
    STDGPU_EXPECTS(capacity > 0);

    float k = static_cast<float>(bucket_count);
    float n = static_cast<float>(capacity);
    index_t result = static_cast<index_t>(n * (1.0 - std::pow(1.0 - (1.0 / k), n - 1.0)));

    STDGPU_ENSURES(result >= 0);

    return result;
}


inline float
default_max_load_factor()
{
    return 1.0f;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::allocator_type
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::get_allocator() const
{
    return allocator_type();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::begin()
{
    return _values;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::begin() const
{
    return _values;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::cbegin() const
{
    return begin();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::end()
{
    return _values + total_count();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::end() const
{
    return _values + total_count();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::cend() const
{
    return end();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct unordered_base_collect_positions
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    unordered_base_collect_positions(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        if (base.occupied(i))
        {
            base._range_indices.push_back(i);
        }
    }
};


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
device_indexed_range<const typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::device_range() const
{
    _range_indices.clear();

    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(total_count()),
                     unordered_base_collect_positions<Key, Value, KeyFromValue, Hash, KeyEqual>(*this));

    return device_indexed_range<const value_type>(_range_indices.device_range(), _values);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct offset_inside_range
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    offset_inside_range(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_HOST_DEVICE bool
    operator()(const index_t i) const
    {
        index_t linked_entry = i + base._offsets[i];

        if (linked_entry < 0 || linked_entry >= base.total_count())
        {
            printf("stdgpu::detail::unordered_base : Linked entry out of range : %d -> %d\n", i, linked_entry);
            return false;
        }

        return true;
    }
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline bool
offset_range_valid(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
{
    return thrust::all_of(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(base.total_count()),
                          offset_inside_range<Key, Value, KeyFromValue, Hash, KeyEqual>(base));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct count_visits
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;
    int* flags;

    count_visits(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base,
                 int* flags)
        : base(base),
          flags(flags)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        index_t linked_list = i;

        stdgpu::atomic_ref<int>(flags[linked_list]).fetch_add(1);

        while (base._offsets[linked_list] != 0)
        {
            linked_list += base._offsets[linked_list];

            stdgpu::atomic_ref<int>(flags[linked_list]).fetch_add(1);

            // Prevent potential endless loop and print warning
            if (flags[linked_list] > 1)
            {
                printf("stdgpu::detail::unordered_base : Linked list not unique : %d visited %d times\n", linked_list, flags[linked_list]);
                return;
            }
        }
    }
};

struct less_equal_one
{
    STDGPU_HOST_DEVICE bool
    operator()(const int flag) const
    {
        return flag <= 1;
    }
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline bool
loop_free(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
{
    int* flags = createDeviceArray<int>(base.total_count(), 0);

    thrust::for_each(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(base.bucket_count()),
                     count_visits<Key, Value, KeyFromValue, Hash, KeyEqual>(base, flags));

    bool result = thrust::all_of(device_cbegin(flags), device_cend(flags),
                                 less_equal_one());

    destroyDeviceArray<int>(flags);

    return result;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct value_reachable
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    value_reachable(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const index_t i) const
    {
        if (base.occupied(i))
        {
            auto block = base._key_from_value(base._values[i]);

            if (!base.contains(block))
            {
                printf("stdgpu::detail::unordered_base : Unreachable entry : %d\n", i);
                return false;
            }
        }

        return true;
    }
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline bool
values_reachable(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
{
    return thrust::all_of(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(base.total_count()),
                          value_reachable<Key, Value, KeyFromValue, Hash, KeyEqual>(base));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct values_unique
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    values_unique(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const index_t i) const
    {
        if (base.occupied(i))
        {
            auto block = base._key_from_value(base._values[i]);

            auto it = base.find(block);
            index_t position = thrust::distance(base.begin(), it);

            if (position != i)
            {
                printf("stdgpu::detail::unordered_base : Duplicate entry : Expected %d but also found at %d\n", i, position);
                return false;
            }
        }

        return true;
    }
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline bool
unique(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
{
    return thrust::all_of(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(base.total_count()),
                          values_unique<Key, Value, KeyFromValue, Hash, KeyEqual>(base));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline bool
occupied_count_valid(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
{
    index_t size_count = base.size();
    index_t size_sum   = base._occupied.count();

    return (size_count == size_sum);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct erase_from_key
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    erase_from_key(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const Key& key)
    {
        base.erase(key);
    }
};


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
struct erase_from_value
{
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> base;

    erase_from_value(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& base)
        : base(base)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const Value& value)
    {
        base.erase(base._key_from_value(value));
    }
};


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::bucket(const key_type& key) const
{
    #if STDGPU_USE_FIBONACCI_HASHING
        // If bucket_count() == 1, then the result will be shifted by the width of std::size_t which leads to undefined/unreliable behavior
        std::size_t result = (bucket_count() == 1) ? 0 : (_hash(key) * 11400714819323198485llu) >> (numeric_limits<std::size_t>::digits - log2pow2<std::size_t>(bucket_count()));
    #else
        std::size_t result = mod2<std::size_t>(_hash(key), bucket_count());
    #endif

    STDGPU_ENSURES(0 <= static_cast<index_t>(result));
    STDGPU_ENSURES(static_cast<index_t>(result) < bucket_count());
    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::bucket_size(index_t n) const
{
    STDGPU_EXPECTS(n < bucket_count());

    index_t result    = 0;
    index_t key_index = n;

    // Bucket
    if (occupied(key_index))
    {
        result++;
    }

    // Linked list
    while (_offsets[key_index] != 0)
    {
        key_index += _offsets[key_index];

        if (occupied(key_index))
        {
            result++;
        }
    }

    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::count(const key_type& key) const
{
    return contains(key) ? index_t(1) : index_t(0);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::find(const key_type& key)
{
    index_t key_index = bucket(key);

    // Bucket
    if (occupied(key_index)
     && _key_equal(_key_from_value(_values[key_index]), key))
    {
        STDGPU_ENSURES(0 <= key_index);
        STDGPU_ENSURES(key_index < total_count());
        return _values + key_index;
    }

    // Linked list
    while (_offsets[key_index] != 0)
    {
        key_index += _offsets[key_index];

        if (occupied(key_index)
         && _key_equal(_key_from_value(_values[key_index]), key))
        {
            STDGPU_ENSURES(0 <= key_index);
            STDGPU_ENSURES(key_index < total_count());
            return _values + key_index;
        }
    }

    return end();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::find(const key_type& key) const
{
    index_t key_index = bucket(key);

    // Bucket
    if (occupied(key_index)
     && _key_equal(_key_from_value(_values[key_index]), key))
    {
        STDGPU_ENSURES(0 <= key_index);
        STDGPU_ENSURES(key_index < total_count());
        return _values + key_index;
    }

    // Linked list
    while (_offsets[key_index] != 0)
    {
        key_index += _offsets[key_index];

        if (occupied(key_index)
         && _key_equal(_key_from_value(_values[key_index]), key))
        {
            STDGPU_ENSURES(0 <= key_index);
            STDGPU_ENSURES(key_index < total_count());
            return _values + key_index;
        }
    }

    return end();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::contains(const key_type& key) const
{
    return find(key) != end();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY thrust::pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator, bool>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::try_insert(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type& value)
{
    iterator inserted_it = end();
    bool inserted = false;

    key_type block = _key_from_value(value);

    if (!contains(block))
    {
        index_t bucket_index = bucket(block);

        // Bucket
        if (!occupied(bucket_index))
        {
            if (_locks[bucket_index].try_lock())
            {
                // START --- critical section --- START

                // !!! VERIFY CONDITIONS HAVE NOT CHANGED !!!
                if (!contains(block) && !occupied(bucket_index))
                {
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::construct(a, &(_values[bucket_index]), value);
                    // Do not touch the linked list
                    //_offsets[bucket_index] = 0;

                    // Set occupied status after entry has been fully constructed
                    ++_occupied_count;
                    bool was_occupied = _occupied.set(bucket_index);

                    inserted_it = begin() + bucket_index;
                    inserted = true;

                    if (was_occupied)
                    {
                        printf("unordered_base::try_insert : Expected entry to be not occupied but actually was\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[bucket_index].unlock();
            }
        }
        // Linked list
        else
        {
            index_t linked_list_end = find_linked_list_end(bucket_index);

            if (_locks[linked_list_end].try_lock())
            {
                // START --- critical section --- START

                // !!! VERIFY CONDITIONS HAVE NOT CHANGED !!!
                index_t checked_linked_list_end = find_linked_list_end(bucket_index);
                if (!contains(block) && linked_list_end == checked_linked_list_end)
                {
                    thrust::pair<index_t, bool> popped = _excess_list_positions.pop_back();

                    if (!popped.second)
                    {
                        printf("unordered_base::try_insert : Associated bucket and excess list full\n");
                    }
                    else
                    {
                        index_t new_linked_list_end = popped.first;

                        allocator_type a = get_allocator();     // Will be replaced by member
                        allocator_traits<allocator_type>::construct(a, &(_values[new_linked_list_end]), value);
                        _offsets[new_linked_list_end] = 0;

                        // Set occupied status after entry has been fully constructed
                        ++_occupied_count;
                        bool was_occupied = _occupied.set(new_linked_list_end);

                        // Connect new linked list end after its values have been fully initialized and the occupied status has been set as try_erase is not resetting offsets
                        _offsets[linked_list_end] = new_linked_list_end - linked_list_end;

                        inserted_it = begin() + new_linked_list_end;
                        inserted = true;

                        if (was_occupied)
                        {
                            printf("unordered_base::try_insert : Expected entry to be not occupied but actually was\n");
                        }
                    }
                }

                //  END  --- critical section ---  END
                _locks[linked_list_end].unlock();
            }
        }
    }

    return thrust::make_pair(inserted_it, inserted);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::try_erase(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type& key)
{
    bool erased = false;

    const_iterator it = find(key);
    index_t position = thrust::distance(cbegin(), it);

    bool contains_block = (it != end());
    if (contains_block)
    {
        index_t bucket_index = bucket(key);

        // Bucket
        if (position == bucket_index)
        {
            if (_locks[position].try_lock())
            {
                // START --- critical section --- START

                // !!! VERIFY CONDITIONS HAVE NOT CHANGED !!!
                const_iterator checked_it = find(key);
                if (it == checked_it)
                {
                    // Set not-occupied status before entry has been fully erased
                    bool was_occupied = _occupied.reset(position);
                    --_occupied_count;

                    // Default values
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::destroy(a, &(_values[position]));
                    // Do not touch the linked list
                    //_offsets[position] = 0;

                    erased = true;

                    if (!was_occupied)
                    {
                        printf("unordered_base::try_erase : Expected entry to be occupied but actually was not\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[position].unlock();
            }
        }
        // Linked list
        else
        {
            index_t previous_position = find_previous_entry_position(position, bucket_index);

            if (try_lock(_locks[position], _locks[previous_position]) == -1)
            {
                // START --- critical section --- START

                // !!! VERIFY CONDITIONS HAVE NOT CHANGED !!!
                const_iterator checked_it = find(key);
                index_t checked_previous_position = find_previous_entry_position(position, bucket_index);
                if (it == checked_it
                 && previous_position == checked_previous_position)
                {
                    // Set offset
                    if (_offsets[position] != 0)
                    {
                        _offsets[previous_position] += _offsets[position];
                    }
                    else
                    {
                        _offsets[previous_position] = 0;
                    }

                    // Set not-occupied status before entry has been fully erased
                    bool was_occupied = _occupied.reset(position);
                    --_occupied_count;

                    // Default values
                    allocator_type a = get_allocator();     // Will be replaced by member
                    allocator_traits<allocator_type>::destroy(a, &(_values[position]));
                    // Do not reset the offset of the erased linked list entry as another thread executing find() might still need it, so make try_insert responsible for resetting it
                    //_offsets[position] = 0;
                    _excess_list_positions.push_back(position);

                    erased = true;

                    if (!was_occupied)
                    {
                        printf("unordered_base::try_erase : Expected entry to be occupied but actually was not\n");
                    }
                }

                //  END  --- critical section ---  END
                _locks[position].unlock();
                _locks[previous_position].unlock();
            }

        }
    }

    return static_cast<index_t>(erased);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::find_linked_list_end(const index_t linked_list_start)
{
    index_t linked_list_end = linked_list_start;

    while (_offsets[linked_list_end] != 0)
    {
        linked_list_end += _offsets[linked_list_end];
    }

    return linked_list_end;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::find_previous_entry_position(const index_t entry_position,
                                                                                       const index_t linked_list_start)
{
    bool position_found = false;
    index_t previous_position = linked_list_start;
    index_t key_index = linked_list_start;

    while (_offsets[key_index] != 0)
    {
        // Next entry
        key_index += _offsets[key_index];
        position_found = (key_index == entry_position);

        if (position_found)
        {
            return previous_position;
        }
        // Increment previous (--> equal to key_index)
        else
        {
            previous_position = key_index;
        }
    }

    return key_index;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
template <class... Args>
inline STDGPU_DEVICE_ONLY thrust::pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator, bool>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::emplace(Args&&... args)
{
    return insert(value_type(forward<Args>(args)...));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY thrust::pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::iterator, bool>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::insert(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type& value)
{
    thrust::pair<iterator, bool> result = thrust::make_pair(end(), false);

    while (true)
    {
        if (!contains(_key_from_value(value))
            && !full() && !_excess_list_positions.empty())
        {
            result = try_insert(value);
        }
        else
        {
            break;
        }
    }

    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::insert(device_ptr<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type> begin,
                                                                 device_ptr<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type> end)
{
    thrust::copy(begin, end,
                 stdgpu::inserter(*this));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::insert(device_ptr<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type> begin,
                                                                 device_ptr<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::value_type> end)
{
    thrust::copy(begin, end,
                 stdgpu::inserter(*this));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::erase(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type& key)
{
    index_t result = 0;

    while (true)
    {
        if (contains(key))
        {
            result = try_erase(key);
        }
        else
        {
            break;
        }
    }

    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::erase(device_ptr<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type> begin,
                                                                device_ptr<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type> end)
{
    thrust::for_each(begin, end,
                     erase_from_key<Key, Value, KeyFromValue, Hash, KeyEqual>(*this));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::erase(device_ptr<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type> begin,
                                                                device_ptr<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_type> end)
{
    thrust::for_each(begin, end,
                     erase_from_key<Key, Value, KeyFromValue, Hash, KeyEqual>(*this));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::occupied(const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < total_count());

    return _occupied[n];
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::empty() const
{
    return (size() == 0);
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::full() const
{
    return (size() == total_count());
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::size() const
{
    index_t current_size = _occupied_count.load();

    STDGPU_ENSURES(0 <= current_size);
    STDGPU_ENSURES(current_size <= total_count());
    return current_size;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::max_size() const
{
    return total_count();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::bucket_count() const
{
    return _bucket_count;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::excess_count() const
{
    return _excess_count;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::total_count() const
{
    return (bucket_count() + excess_count());
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE float
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::load_factor() const
{
    return static_cast<float>(size()) / static_cast<float>(bucket_count());
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE float
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::max_load_factor() const
{
    return default_max_load_factor();
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::hasher
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::hash_function() const
{
    return _hash;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_equal
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::key_eq() const
{
    return _key_equal;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::valid() const
{
    // Special case : Zero capacity is valid
    if (total_count() == 0) return true;


    return (offset_range_valid(*this)
         && loop_free(*this)
         && values_reachable(*this)
         && unique(*this)
         && occupied_count_valid(*this)
         && _locks.valid()
         && _excess_list_positions.valid());
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::clear()
{
    auto range = device_range();
    thrust::for_each(range.begin(), range.end(),
                     erase_from_value<Key, Value, KeyFromValue, Hash, KeyEqual>(*this));
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::createDeviceObject(const index_t& capacity)
{
    STDGPU_EXPECTS(capacity > 0);

    // bucket count depends on default max load factor
    index_t bucket_count = next_pow2(std::ceil(static_cast<float>(capacity) / default_max_load_factor()));

    // excess count is estimated by the expected collision count and conservatively lowered since entries falling into regular buckets are already included here
    index_t excess_count = std::max<index_t>(1, expected_collisions(bucket_count, capacity) * 2 / 3);

    index_t total_count = bucket_count + excess_count;

    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> result;
    allocator_type a;   // Will be replaced by member
    result._bucket_count            = bucket_count;
    result._excess_count            = excess_count;
    result._values                  = allocator_traits<allocator_type>::allocate(a, total_count);
    result._offsets                 = createDeviceArray<index_t>(total_count, 0);
    result._occupied                = bitset::createDeviceObject(total_count);
    result._occupied_count          = atomic<int>::createDeviceObject();
    result._locks                   = mutex_array::createDeviceObject(total_count);
    result._excess_list_positions   = vector<index_t>::createDeviceObject(excess_count);
    result._key_from_value          = key_from_value();
    result._hash                    = hasher();
    result._key_equal               = key_equal();

    result._range_indices           = vector<index_t>::createDeviceObject(total_count);

    thrust::copy(thrust::device,
                 thrust::counting_iterator<index_t>(bucket_count), thrust::counting_iterator<index_t>(bucket_count + excess_count),
                 stdgpu::back_inserter(result._excess_list_positions));

    STDGPU_ENSURES(result._excess_list_positions.full());

    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::createDeviceObject(const index_t& bucket_count,
                                                                             const index_t& excess_count)
{
    STDGPU_EXPECTS(bucket_count > 0);
    STDGPU_EXPECTS(excess_count > 0);
    STDGPU_EXPECTS(ispow2<std::size_t>(bucket_count));

    index_t total_count = bucket_count + excess_count;

    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual> result;
    allocator_type a;   // Will be replaced by member
    result._bucket_count            = bucket_count;
    result._excess_count            = excess_count;
    result._values                  = allocator_traits<allocator_type>::allocate(total_count);
    result._offsets                 = createDeviceArray<index_t>(total_count, 0);
    result._occupied                = bitset::createDeviceObject(total_count);
    result._occupied_count          = atomic<int>::createDeviceObject();
    result._locks                   = mutex_array::createDeviceObject(total_count);
    result._excess_list_positions   = vector<index_t>::createDeviceObject(excess_count);
    result._key_from_value          = key_from_value();
    result._hash                    = hasher();
    result._key_equal               = key_equal();

    result._range_indices           = vector<index_t>::createDeviceObject(total_count);

    thrust::copy(thrust::device,
                 thrust::counting_iterator<index_t>(bucket_count), thrust::counting_iterator<index_t>(bucket_count + excess_count),
                 stdgpu::back_inserter(result._excess_list_positions));

    STDGPU_ENSURES(result._excess_list_positions.full());

    return result;
}


template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual>
void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>::destroyDeviceObject(unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual>& device_object)
{
    device_object.clear();

    allocator_type a = device_object.get_allocator();   // Will be replaced by member
    index_t total_count = device_object._bucket_count + device_object._excess_count;
    allocator_traits<allocator_type>::deallocate(a, device_object._values, total_count);

    device_object._bucket_count = 0;
    device_object._excess_count = 0;
    destroyDeviceArray<index_t>(device_object._offsets);
    bitset::destroyDeviceObject(device_object._occupied);
    atomic<int>::destroyDeviceObject(device_object._occupied_count);
    mutex_array::destroyDeviceObject(device_object._locks);
    vector<index_t>::destroyDeviceObject(device_object._excess_list_positions);
    device_object._key_from_value   = key_from_value();
    device_object._hash             = hasher();
    device_object._key_equal        = key_equal();

    vector<index_t>::destroyDeviceObject(device_object._range_indices);
}

} // namespace detail

} // namespace stdgpu



#endif // STDGPU_UNORDERED_BASE_DETAIL_H
