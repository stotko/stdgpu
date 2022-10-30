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

#ifndef STDGPU_UNORDERED_MAP_DETAIL_H
#define STDGPU_UNORDERED_MAP_DETAIL_H

#include <utility>

#include <stdgpu/bit.h>
#include <stdgpu/contract.h>
#include <stdgpu/utility.h>

namespace stdgpu
{

namespace detail
{

template <typename Pair>
struct select1st
{
    STDGPU_HOST_DEVICE typename Pair::first_type
    operator()(const Pair& pair) const
    {
        return pair.first;
    }
};

} // namespace detail

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::allocator_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::get_allocator() const noexcept
{
    return _base.get_allocator();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::begin() noexcept
{
    return _base.begin();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::begin() const noexcept
{
    return _base.begin();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::cbegin() const noexcept
{
    return _base.cbegin();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::end() noexcept
{
    return _base.end();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::end() const noexcept
{
    return _base.end();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::cend() const noexcept
{
    return _base.cend();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
device_indexed_range<const typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::value_type>
unordered_map<Key, T, Hash, KeyEqual, Allocator>::device_range() const
{
    return _base.device_range();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::index_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::bucket(const key_type& key) const
{
    return _base.bucket(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::index_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::bucket_size(index_type n) const
{
    return _base.bucket_size(n);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::index_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::count(const key_type& key) const
{
    return _base.count(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::index_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::count(const KeyLike& key) const
{
    return _base.count(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::find(const key_type& key)
{
    return _base.find(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::find(const key_type& key) const
{
    return _base.find(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::find(const KeyLike& key)
{
    return _base.find(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::const_iterator
unordered_map<Key, T, Hash, KeyEqual, Allocator>::find(const KeyLike& key) const
{
    return _base.find(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
unordered_map<Key, T, Hash, KeyEqual, Allocator>::contains(const key_type& key) const
{
    return _base.contains(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyLike,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_transparent_v<Hash>&& detail::is_transparent_v<KeyEqual>)>
inline STDGPU_DEVICE_ONLY bool
unordered_map<Key, T, Hash, KeyEqual, Allocator>::contains(const KeyLike& key) const
{
    return _base.contains(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <class... Args>
inline STDGPU_DEVICE_ONLY pair<typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator, bool>
unordered_map<Key, T, Hash, KeyEqual, Allocator>::emplace(Args&&... args)
{
    return _base.emplace(forward<Args>(args)...);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY pair<typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator, bool>
unordered_map<Key, T, Hash, KeyEqual, Allocator>::insert(
        const unordered_map<Key, T, Hash, KeyEqual, Allocator>::value_type& value)
{
    return _base.insert(value);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename ValueIterator, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_iterator_v<ValueIterator>)>
inline void
unordered_map<Key, T, Hash, KeyEqual, Allocator>::insert(ValueIterator begin, ValueIterator end)
{
    _base.insert(begin, end);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::index_type
unordered_map<Key, T, Hash, KeyEqual, Allocator>::erase(
        const unordered_map<Key, T, Hash, KeyEqual, Allocator>::key_type& key)
{
    return _base.erase(key);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
template <typename KeyIterator, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::is_iterator_v<KeyIterator>)>
inline void
unordered_map<Key, T, Hash, KeyEqual, Allocator>::erase(KeyIterator begin, KeyIterator end)
{
    _base.erase(begin, end);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE bool
unordered_map<Key, T, Hash, KeyEqual, Allocator>::empty() const
{
    return _base.empty();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE bool
unordered_map<Key, T, Hash, KeyEqual, Allocator>::full() const
{
    return _base.full();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual, Allocator>::size() const
{
    return _base.size();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual, Allocator>::max_size() const noexcept
{
    return _base.max_size();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual, Allocator>::bucket_count() const
{
    return _base.bucket_count();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE float
unordered_map<Key, T, Hash, KeyEqual, Allocator>::load_factor() const
{
    return _base.load_factor();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE float
unordered_map<Key, T, Hash, KeyEqual, Allocator>::max_load_factor() const
{
    return _base.max_load_factor();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::hasher
unordered_map<Key, T, Hash, KeyEqual, Allocator>::hash_function() const
{
    return _base.hash_function();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual, Allocator>::key_equal
unordered_map<Key, T, Hash, KeyEqual, Allocator>::key_eq() const
{
    return _base.key_eq();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
bool
unordered_map<Key, T, Hash, KeyEqual, Allocator>::valid() const
{
    return _base.valid();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void
unordered_map<Key, T, Hash, KeyEqual, Allocator>::clear()
{
    _base.clear();
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
unordered_map<Key, T, Hash, KeyEqual, Allocator>
unordered_map<Key, T, Hash, KeyEqual, Allocator>::createDeviceObject(const index_t& capacity,
                                                                     const Allocator& allocator)
{
    STDGPU_EXPECTS(capacity > 0);

    unordered_map<Key, T, Hash, KeyEqual, Allocator> result(base_type::createDeviceObject(capacity, allocator));

    return result;
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void
unordered_map<Key, T, Hash, KeyEqual, Allocator>::destroyDeviceObject(
        unordered_map<Key, T, Hash, KeyEqual, Allocator>& device_object)
{
    base_type::destroyDeviceObject(device_object._base);
}

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
unordered_map<Key, T, Hash, KeyEqual, Allocator>::unordered_map(base_type&& base)
  : _base(std::move(base))
{
}

} // namespace stdgpu

#endif // STDGPU_UNORDERED_MAP_DETAIL_H
