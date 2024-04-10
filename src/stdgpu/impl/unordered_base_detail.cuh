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

#include <cmath>

#include <stdgpu/algorithm.h>
#include <stdgpu/bit.h>
#include <stdgpu/contract.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/utility.h>

namespace stdgpu::detail
{

inline index_t
expected_collisions(const index_t bucket_count, const index_t capacity)
{
    STDGPU_EXPECTS(bucket_count > 0);
    STDGPU_EXPECTS(capacity > 0);

    long double k = static_cast<long double>(bucket_count);
    long double n = static_cast<long double>(capacity);
    // NOLINTNEXTLINE(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    index_t result = static_cast<index_t>(std::ceil(n * (1.0L - std::pow(1.0L - (1.0L / k), n - 1.0L))));

    STDGPU_ENSURES(result >= 0);

    return result;
}

inline STDGPU_HOST_DEVICE float
default_max_load_factor()
{
    return 1.0F;
}

inline STDGPU_HOST_DEVICE index_t
fibonacci_hashing(const std::size_t hash, const index_t bucket_count)
{
    index_t max_bit_width_result =
            static_cast<index_t>(bit_width<std::size_t>(static_cast<std::size_t>(bucket_count)) - 1);

    // Resulting index will always be zero, but shift by the width of std::size_t is undefined/unreliable behavior, so
    // handle this special case
    if (max_bit_width_result <= 0)
    {
        return 0;
    }

    const std::size_t dropped_bit_width =
            static_cast<std::size_t>(numeric_limits<std::size_t>::digits - max_bit_width_result);

    // Improve robustness for Multiplicative Hashing
    const std::size_t improved_hash = hash ^ (hash >> dropped_bit_width);

    // 2^64/phi, where phi is the golden ratio
    const std::size_t multiplier = 11400714819323198485LLU;

    // Multiplicative Hashing to the desired range
    index_t result = static_cast<index_t>((multiplier * improved_hash) >> dropped_bit_width);

    STDGPU_ENSURES(0 <= result);
    STDGPU_ENSURES(result < bucket_count);

    return result;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::allocator_type
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::get_allocator() const noexcept
{
    return _allocator;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::begin() noexcept
{
    return _values;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::begin() const noexcept
{
    return _values;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::cbegin() const noexcept
{
    return begin();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::end() noexcept
{
    return _values + total_count();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::end() const noexcept
{
    return _values + total_count();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::cend() const noexcept
{
    return end();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class unordered_base_collect_positions
{
public:
    explicit unordered_base_collect_positions(
            const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
      : _base(base)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        if (_base.occupied(i))
        {
            index_t j = _base._range_indices_end++;
            _base._range_indices[j] = i;
        }
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
device_indexed_range<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::device_range()
{
    return device_range(execution::device);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
device_indexed_range<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::device_range(ExecutionPolicy&& policy)
{
    _range_indices_end.store(0);

    for_each_index(std::forward<ExecutionPolicy>(policy),
                   total_count(),
                   unordered_base_collect_positions<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(*this));

    return device_indexed_range<value_type>(stdgpu::device_range<index_t>(_range_indices, _range_indices_end.load()),
                                            _values);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
device_indexed_range<const typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::device_range() const
{
    return device_range(execution::device);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
device_indexed_range<const typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::device_range(ExecutionPolicy&& policy) const
{
    _range_indices_end.store(0);

    for_each_index(std::forward<ExecutionPolicy>(policy),
                   total_count(),
                   unordered_base_collect_positions<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(*this));

    return device_indexed_range<const value_type>(
            stdgpu::device_range<index_t>(_range_indices, _range_indices_end.load()),
            _values);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class offset_inside_range
{
public:
    explicit offset_inside_range(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
      : _base(base)
    {
    }

    STDGPU_HOST_DEVICE bool
    operator()(const index_t i) const
    {
        index_t linked_entry = i + _base._offsets[i];

        if (linked_entry < 0 || linked_entry >= _base.total_count())
        {
            printf("stdgpu::detail::unordered_base : Linked entry out of range : %" STDGPU_PRIINDEX
                   " -> %" STDGPU_PRIINDEX "\n",
                   i,
                   linked_entry);
            return false;
        }

        return true;
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
};

template <typename ExecutionPolicy,
          typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator>
inline bool
offset_range_valid(ExecutionPolicy&& policy,
                   const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
{
    return transform_reduce_index(std::forward<ExecutionPolicy>(policy),
                                  base.total_count(),
                                  true,
                                  logical_and<>(),
                                  offset_inside_range<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(base));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class count_visits
{
public:
    count_visits(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base, int* flags)
      : _base(base)
      , _flags(flags)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        index_t linked_list = i;

        atomic_ref<int>(_flags[linked_list]).fetch_add(1);

        while (_base._offsets[linked_list] != 0)
        {
            linked_list += _base._offsets[linked_list];

            atomic_ref<int>(_flags[linked_list]).fetch_add(1);

            // Prevent potential endless loop and print warning
            if (_flags[linked_list] > 1)
            {
                printf("stdgpu::detail::unordered_base : Linked list not unique : %" STDGPU_PRIINDEX
                       " visited %d times\n",
                       linked_list,
                       _flags[linked_list]);
                return;
            }
        }
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
    int* _flags;
};

class less_equal_one
{
public:
    explicit less_equal_one(int* flags)
      : _flags(flags)
    {
    }

    STDGPU_HOST_DEVICE bool
    operator()(const index_t i) const
    {
        return _flags[i] <= 1;
    }

private:
    int* _flags;
};

template <typename ExecutionPolicy,
          typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator>
inline bool
loop_free(ExecutionPolicy&& policy, const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
{
    int* flags = createDeviceArray<int>(base.total_count(), 0);

    for_each_index(std::forward<ExecutionPolicy>(policy),
                   base.bucket_count(),
                   count_visits<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(base, flags));

    bool result = transform_reduce_index(std::forward<ExecutionPolicy>(policy),
                                         base.total_count(),
                                         true,
                                         logical_and<>(),
                                         less_equal_one(flags));

    destroyDeviceArray<int>(flags);

    return result;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class value_reachable
{
public:
    explicit value_reachable(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
      : _base(base)
    {
    }

    STDGPU_DEVICE_ONLY bool
    operator()(const index_t i) const
    {
        if (_base.occupied(i))
        {
            auto block = _base._key_from_value(_base._values[i]);

            if (!_base.contains(block))
            {
                printf("stdgpu::detail::unordered_base : Unreachable entry : %" STDGPU_PRIINDEX "\n", i);
                return false;
            }
        }

        return true;
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
};

template <typename ExecutionPolicy,
          typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator>
inline bool
values_reachable(ExecutionPolicy&& policy,
                 const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
{
    return transform_reduce_index(std::forward<ExecutionPolicy>(policy),
                                  base.total_count(),
                                  true,
                                  logical_and<>(),
                                  value_reachable<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(base));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class values_unique
{
public:
    explicit values_unique(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
      : _base(base)
    {
    }

    STDGPU_DEVICE_ONLY bool
    operator()(const index_t i) const
    {
        if (_base.occupied(i))
        {
            auto block = _base._key_from_value(_base._values[i]);

            auto it = _base.find(block); // NOLINT(readability-qualified-auto)
            index_t position = static_cast<index_t>(it - _base.begin());

            if (position != i)
            {
                printf("stdgpu::detail::unordered_base : Duplicate entry : Expected %" STDGPU_PRIINDEX
                       " but also found at %" STDGPU_PRIINDEX "\n",
                       i,
                       position);
                return false;
            }
        }

        return true;
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
};

template <typename ExecutionPolicy,
          typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator>
inline bool
unique(ExecutionPolicy&& policy, const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
{
    return transform_reduce_index(std::forward<ExecutionPolicy>(policy),
                                  base.total_count(),
                                  true,
                                  logical_and<>(),
                                  values_unique<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(base));
}

template <typename ExecutionPolicy,
          typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator>
inline bool
occupied_count_valid(ExecutionPolicy&& policy,
                     const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
{
    index_t size_count = base.size();
    index_t size_sum = base._occupied.count(std::forward<ExecutionPolicy>(policy));

    return (size_count == size_sum);
}

template <typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator,
          typename InputIt>
class insert_value
{
public:
    insert_value(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base, InputIt begin)
      : _base(base)
      , _begin(begin)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        _base.insert(*to_address(_begin + i));
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
    InputIt _begin;
};

template <typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual,
          typename Allocator,
          typename KeyIterator>
class erase_from_key
{
public:
    erase_from_key(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base, KeyIterator begin)
      : _base(base)
      , _begin(begin)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t i)
    {
        _base.erase(*(_begin + i));
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
    KeyIterator _begin;
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
class destroy_values
{
public:
    explicit destroy_values(const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& base)
      : _base(base)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const index_t n)
    {
        if (_base.occupied(n))
        {
            allocator_traits<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::
                                     allocator_type>::destroy(_base._allocator, &(_base._values[n]));
        }
    }

private:
    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> _base;
};

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::bucket(const key_type& key) const
{
    return bucket_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::bucket_impl(const KeyLike& key) const
{
    index_t result = fibonacci_hashing(_hash(key), bucket_count());

    STDGPU_ENSURES(0 <= result);
    STDGPU_ENSURES(result < bucket_count());
    return result;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::bucket_size(index_t n) const
{
    STDGPU_EXPECTS(n < bucket_count());

    index_t result = 0;
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

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::count(const key_type& key) const
{
    return count_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::count(const KeyLike& key) const
{
    return count_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::count_impl(const KeyLike& key) const
{
    return contains(key) ? index_t(1) : index_t(0);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find(const key_type& key)
{
    return const_cast<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator>(
            static_cast<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>*>(this)->find(key));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find(const key_type& key) const
{
    return find_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find(const KeyLike& key)
{
    return const_cast<unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator>(
            static_cast<const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>*>(this)->find(key));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find(const KeyLike& key) const
{
    return find_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike>
inline STDGPU_DEVICE_ONLY typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::const_iterator
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find_impl(const KeyLike& key) const
{
    index_t key_index = bucket_impl(key);

    // Bucket
    if (occupied(key_index) && _key_equal(_key_from_value(_values[key_index]), key))
    {
        STDGPU_ENSURES(0 <= key_index);
        STDGPU_ENSURES(key_index < total_count());
        return _values + key_index;
    }

    // Linked list
    while (_offsets[key_index] != 0)
    {
        key_index += _offsets[key_index];

        if (occupied(key_index) && _key_equal(_key_from_value(_values[key_index]), key))
        {
            STDGPU_ENSURES(0 <= key_index);
            STDGPU_ENSURES(key_index < total_count());
            return _values + key_index;
        }
    }

    return end();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::contains(const key_type& key) const
{
    return contains_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::contains(const KeyLike& key) const
{
    return contains_impl(key);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::contains_impl(const KeyLike& key) const
{
    return find(key) != end();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY
        pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator, operation_status>
        unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::try_insert(
                const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type& value)
{
    iterator inserted_it = end();
    operation_status status = operation_status::failed_collision;

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
                    allocator_traits<allocator_type>::construct(_allocator, &(_values[bucket_index]), value);
                    // Do not touch the linked list
                    //_offsets[bucket_index] = 0;

                    // Set occupied status after entry has been fully constructed
                    ++_occupied_count;
                    bool was_occupied = _occupied.set(bucket_index);

                    inserted_it = begin() + bucket_index;
                    status = operation_status::success;

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
                    pair<index_t, bool> popped = _excess_list_positions.pop_back();

                    if (!popped.second)
                    {
                        printf("unordered_base::try_insert : Associated bucket and excess list full\n");
                    }
                    else
                    {
                        index_t new_linked_list_end = popped.first;

                        allocator_traits<allocator_type>::construct(_allocator, &(_values[new_linked_list_end]), value);
                        _offsets[new_linked_list_end] = 0;

                        // Set occupied status after entry has been fully constructed
                        ++_occupied_count;
                        bool was_occupied = _occupied.set(new_linked_list_end);

                        // Connect new linked list end after its values have been fully initialized and the occupied
                        // status has been set as try_erase is not resetting offsets
                        _offsets[linked_list_end] = new_linked_list_end - linked_list_end;

                        inserted_it = begin() + new_linked_list_end;
                        status = operation_status::success;

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
    else
    {
        status = operation_status::failed_no_action_required;
    }

    return pair<iterator, operation_status>(inserted_it, status);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY operation_status
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::try_erase(
        const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::key_type& key)
{
    operation_status status = operation_status::failed_collision;

    const_iterator it = find(key);
    index_t position = static_cast<index_t>(it - cbegin());

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
                    allocator_traits<allocator_type>::destroy(_allocator, &(_values[position]));
                    // Do not touch the linked list
                    //_offsets[position] = 0;

                    status = operation_status::success;

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
                if (it == checked_it && previous_position == checked_previous_position)
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
                    allocator_traits<allocator_type>::destroy(_allocator, &(_values[position]));
                    // Do not reset the offset of the erased linked list entry as another thread executing find() might
                    // still need it, so make try_insert responsible for resetting it
                    //_offsets[position] = 0;
                    _excess_list_positions.push_back(position);

                    status = operation_status::success;

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
    else
    {
        status = operation_status::failed_no_action_required;
    }

    return status;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find_linked_list_end(
        const index_t linked_list_start)
{
    index_t linked_list_end = linked_list_start;

    while (_offsets[linked_list_end] != 0)
    {
        linked_list_end += _offsets[linked_list_end];
    }

    return linked_list_end;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::find_previous_entry_position(
        const index_t entry_position,
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
            break;
        }

        // Increment previous (--> equal to key_index)
        previous_position = key_index;
    }

    return previous_position;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <class... Args>
inline STDGPU_DEVICE_ONLY
        pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator, bool>
        unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::emplace(Args&&... args)
{
    return insert(value_type(forward<Args>(args)...));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY
        pair<typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::iterator, bool>
        unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::insert(
                const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::value_type& value)
{
    pair<iterator, operation_status> result(end(), operation_status::failed_collision);

    while (true)
    {
        if (result.second == operation_status::failed_collision && !full() && !_excess_list_positions.empty())
        {
            result = try_insert(value);
        }
        else
        {
            break;
        }
    }

    return result.second == operation_status::success ? pair<iterator, bool>(result.first, true)
                                                      : pair<iterator, bool>(result.first, false);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename InputIt, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_iterator_v<InputIt>)>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::insert(InputIt begin, InputIt end)
{
    insert(execution::device, begin, end);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          typename InputIt,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(
                  is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>&& detail::is_iterator_v<InputIt>)>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::insert(ExecutionPolicy&& policy,
                                                                            InputIt begin,
                                                                            InputIt end)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   static_cast<index_t>(end - begin),
                   insert_value<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator, InputIt>(*this, begin));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::erase(
        const unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::key_type& key)
{
    operation_status result = operation_status::failed_collision;

    while (true)
    {
        if (result == operation_status::failed_collision)
        {
            result = try_erase(key);
        }
        else
        {
            break;
        }
    }

    return result == operation_status::success ? 1 : 0;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyIterator, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_iterator_v<KeyIterator>)>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::erase(KeyIterator begin, KeyIterator end)
{
    erase(execution::device, begin, end);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          typename KeyIterator,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(
                  is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>&& detail::is_iterator_v<KeyIterator>)>
inline void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::erase(ExecutionPolicy&& policy,
                                                                           KeyIterator begin,
                                                                           KeyIterator end)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   static_cast<index_t>(end - begin),
                   erase_from_key<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator, KeyIterator>(*this, begin));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::occupied(const index_t n) const
{
    STDGPU_EXPECTS(0 <= n);
    STDGPU_EXPECTS(n < total_count());

    return _occupied[n];
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::empty() const
{
    return (size() == 0);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::full() const
{
    return (size() == total_count());
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::size() const
{
    index_t current_size = _occupied_count.load();

    STDGPU_ENSURES(0 <= current_size);
    STDGPU_ENSURES(current_size <= total_count());
    return current_size;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::max_size() const noexcept
{
    return total_count();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::bucket_count() const
{
    return _bucket_count;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::total_count() const noexcept
{
    return _occupied.size();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE float
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::load_factor() const
{
    return static_cast<float>(size()) / static_cast<float>(bucket_count());
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE float
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::max_load_factor() const
{
    return default_max_load_factor();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::hasher
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::hash_function() const
{
    return _hash;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::key_equal
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::key_eq() const
{
    return _key_equal;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::valid() const
{
    return valid(execution::device);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
bool
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::valid(ExecutionPolicy&& policy) const
{
    // Special case : Zero capacity is valid
    if (total_count() == 0)
    {
        return true;
    }

    return (offset_range_valid(std::forward<ExecutionPolicy>(policy), *this) &&
            loop_free(std::forward<ExecutionPolicy>(policy), *this) &&
            values_reachable(std::forward<ExecutionPolicy>(policy), *this) &&
            unique(std::forward<ExecutionPolicy>(policy), *this) &&
            occupied_count_valid(std::forward<ExecutionPolicy>(policy), *this) &&
            _locks.valid(std::forward<ExecutionPolicy>(policy)) &&
            _excess_list_positions.valid(std::forward<ExecutionPolicy>(policy)));
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::clear()
{
    clear(execution::device);
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::clear(ExecutionPolicy&& policy)
{
    if (empty())
    {
        return;
    }

    if (!detail::is_destroy_optimizable<Value>())
    {
        for_each_index(std::forward<ExecutionPolicy>(policy),
                       total_count(),
                       destroy_values<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>(*this));
    }

    fill(std::forward<ExecutionPolicy>(policy), device_begin(_offsets), device_end(_offsets), 0);

    _occupied.reset(std::forward<ExecutionPolicy>(policy));

    _occupied_count.store(0);

    detail::vector_clear_iota(std::forward<ExecutionPolicy>(policy), _excess_list_positions, bucket_count());
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::createDeviceObject(ExecutionPolicy&& policy,
                                                                                        const index_t& capacity,
                                                                                        const Allocator& allocator)
{
    STDGPU_EXPECTS(capacity > 0);

    // bucket count depends on default max load factor
    index_t bucket_count = static_cast<index_t>(
            bit_ceil(static_cast<std::size_t>(std::ceil(static_cast<float>(capacity) / default_max_load_factor()))));

    // excess count is estimated by the expected collision count:
    // - Conservatively lower the amount since entries falling into regular buckets are already included here
    // - Increase amount by 1 since insertion expects a non-empty excess list also in case of no collision
    index_t excess_count = expected_collisions(bucket_count, capacity) * 2 / 3 + 1;

    index_t total_count = bucket_count + excess_count;

    unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator> result(
            bitset<bitset_default_type, bitset_allocator_type>::createDeviceObject(
                    std::forward<ExecutionPolicy>(policy),
                    total_count,
                    bitset_allocator_type(allocator)),
            atomic<int, atomic_allocator_type>::createDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                                   atomic_allocator_type(allocator)),
            vector<index_t, index_allocator_type>::createDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                                      excess_count,
                                                                      index_allocator_type(allocator)),
            mutex_array<mutex_default_type, mutex_array_allocator_type>::createDeviceObject(
                    std::forward<ExecutionPolicy>(policy),
                    total_count,
                    mutex_array_allocator_type(allocator)),
            atomic<int, atomic_allocator_type>::createDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                                   atomic_allocator_type(allocator)),
            allocator);
    result._bucket_count = bucket_count;
    result._values = allocator_traits<allocator_type>::allocate(result._allocator, total_count);
    result._offsets = allocator_traits<index_allocator_type>::allocate_filled(std::forward<ExecutionPolicy>(policy),
                                                                              result._index_allocator,
                                                                              total_count,
                                                                              0);
    result._range_indices = allocator_traits<index_allocator_type>::allocate(result._index_allocator, total_count);
    result._key_from_value = key_from_value();
    result._hash = hasher();
    result._key_equal = key_equal();

    detail::vector_clear_iota(std::forward<ExecutionPolicy>(policy), result._excess_list_positions, bucket_count);

    STDGPU_ENSURES(result._excess_list_positions.full());

    return result;
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::destroyDeviceObject(
        ExecutionPolicy&& policy,
        unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>& device_object)
{
    if (!detail::is_destroy_optimizable<value_type>())
    {
        device_object.clear();
    }

    device_object._bucket_count = 0;
    allocator_traits<allocator_type>::deallocate(device_object._allocator,
                                                 device_object._values,
                                                 device_object.total_count());
    allocator_traits<index_allocator_type>::deallocate_filled(std::forward<ExecutionPolicy>(policy),
                                                              device_object._index_allocator,
                                                              device_object._offsets,
                                                              device_object.total_count());
    allocator_traits<index_allocator_type>::deallocate(device_object._index_allocator,
                                                       device_object._range_indices,
                                                       device_object.total_count());
    bitset<bitset_default_type, bitset_allocator_type>::destroyDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                                            device_object._occupied);
    atomic<int, atomic_allocator_type>::destroyDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                            device_object._occupied_count);
    mutex_array<mutex_default_type, mutex_array_allocator_type>::destroyDeviceObject(
            std::forward<ExecutionPolicy>(policy),
            device_object._locks);
    vector<index_t, index_allocator_type>::destroyDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                               device_object._excess_list_positions);
    atomic<int, atomic_allocator_type>::destroyDeviceObject(std::forward<ExecutionPolicy>(policy),
                                                            device_object._range_indices_end);
    device_object._key_from_value = key_from_value();
    device_object._hash = hasher();
    device_object._key_equal = key_equal();
}

template <typename Key, typename Value, typename KeyFromValue, typename Hash, typename KeyEqual, typename Allocator>
unordered_base<Key, Value, KeyFromValue, Hash, KeyEqual, Allocator>::unordered_base(
        const bitset<bitset_default_type, bitset_allocator_type>& occupied,
        const atomic<int, atomic_allocator_type>& occupied_count,
        const vector<index_t, index_allocator_type>& excess_list_positions,
        const mutex_array<mutex_default_type, mutex_array_allocator_type>& locks,
        const atomic<int, atomic_allocator_type>& range_indices_end,
        const Allocator& allocator)
  : _occupied(occupied)
  , _occupied_count(occupied_count)
  , _excess_list_positions(excess_list_positions)
  , _locks(locks)
  , _range_indices_end(range_indices_end)
  , _allocator(allocator)
  , _index_allocator(allocator)
{
}

} // namespace stdgpu::detail

#endif // STDGPU_UNORDERED_BASE_DETAIL_H
