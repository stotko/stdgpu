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

template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual>::allocator_type
unordered_map<Key, T, Hash, KeyEqual>::get_allocator() const
{
    return _base.get_allocator();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::iterator
unordered_map<Key, T, Hash, KeyEqual>::begin()
{
    return _base.begin();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::const_iterator
unordered_map<Key, T, Hash, KeyEqual>::begin() const
{
    return _base.begin();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::const_iterator
unordered_map<Key, T, Hash, KeyEqual>::cbegin() const
{
    return _base.cbegin();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::iterator
unordered_map<Key, T, Hash, KeyEqual>::end()
{
    return _base.end();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::const_iterator
unordered_map<Key, T, Hash, KeyEqual>::end() const
{
    return _base.end();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::const_iterator
unordered_map<Key, T, Hash, KeyEqual>::cend() const
{
    return _base.cend();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
device_indexed_range<const typename unordered_map<Key, T, Hash, KeyEqual>::value_type>
unordered_map<Key, T, Hash, KeyEqual>::device_range() const
{
    return _base.device_range();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual>::index_type
unordered_map<Key, T, Hash, KeyEqual>::bucket(const key_type& key) const
{
    return _base.bucket(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::index_type
unordered_map<Key, T, Hash, KeyEqual>::bucket_size(index_type n) const
{
    return _base.bucket_size(n);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::index_type
unordered_map<Key, T, Hash, KeyEqual>::count(const key_type& key) const
{
    return _base.count(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::iterator
unordered_map<Key, T, Hash, KeyEqual>::find(const key_type& key)
{
    return _base.find(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::const_iterator
unordered_map<Key, T, Hash, KeyEqual>::find(const key_type& key) const
{
    return _base.find(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY bool
unordered_map<Key, T, Hash, KeyEqual>::contains(const key_type& key) const
{
    return _base.contains(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
template <class... Args>
inline STDGPU_DEVICE_ONLY thrust::pair<typename unordered_map<Key, T, Hash, KeyEqual>::iterator, bool>
unordered_map<Key, T, Hash, KeyEqual>::emplace(Args&&... args)
{
    return _base.emplace(forward<Args>(args)...);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY thrust::pair<typename unordered_map<Key, T, Hash, KeyEqual>::iterator, bool>
unordered_map<Key, T, Hash, KeyEqual>::insert(const unordered_map<Key, T, Hash, KeyEqual>::value_type& value)
{
    return _base.insert(value);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline void
unordered_map<Key, T, Hash, KeyEqual>::insert(device_ptr<unordered_map<Key, T, Hash, KeyEqual>::value_type> begin,
                                              device_ptr<unordered_map<Key, T, Hash, KeyEqual>::value_type> end)
{
    _base.insert(begin, end);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline void
unordered_map<Key, T, Hash, KeyEqual>::insert(device_ptr<const unordered_map<Key, T, Hash, KeyEqual>::value_type> begin,
                                              device_ptr<const unordered_map<Key, T, Hash, KeyEqual>::value_type> end)
{
    _base.insert(begin, end);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_DEVICE_ONLY typename unordered_map<Key, T, Hash, KeyEqual>::index_type
unordered_map<Key, T, Hash, KeyEqual>::erase(const unordered_map<Key, T, Hash, KeyEqual>::key_type& key)
{
    return _base.erase(key);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline void
unordered_map<Key, T, Hash, KeyEqual>::erase(device_ptr<unordered_map<Key, T, Hash, KeyEqual>::key_type> begin,
                                             device_ptr<unordered_map<Key, T, Hash, KeyEqual>::key_type> end)
{
    _base.erase(begin, end);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline void
unordered_map<Key, T, Hash, KeyEqual>::erase(device_ptr<const unordered_map<Key, T, Hash, KeyEqual>::key_type> begin,
                                             device_ptr<const unordered_map<Key, T, Hash, KeyEqual>::key_type> end)
{
    _base.erase(begin, end);
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE bool
unordered_map<Key, T, Hash, KeyEqual>::empty() const
{
    return _base.empty();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE bool
unordered_map<Key, T, Hash, KeyEqual>::full() const
{
    return _base.full();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual>::size() const
{
    return _base.size();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual>::max_size() const
{
    return _base.max_size();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual>::bucket_count() const
{
    return _base.bucket_count();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual>::excess_count() const
{
    return _base.excess_count();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE index_t
unordered_map<Key, T, Hash, KeyEqual>::total_count() const
{
    return _base.total_count();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE float
unordered_map<Key, T, Hash, KeyEqual>::load_factor() const
{
    return _base.load_factor();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE float
unordered_map<Key, T, Hash, KeyEqual>::max_load_factor() const
{
    return _base.max_load_factor();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual>::hasher
unordered_map<Key, T, Hash, KeyEqual>::hash_function() const
{
    return _base.hash_function();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
inline STDGPU_HOST_DEVICE typename unordered_map<Key, T, Hash, KeyEqual>::key_equal
unordered_map<Key, T, Hash, KeyEqual>::key_eq() const
{
    return _base.key_eq();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
bool
unordered_map<Key, T, Hash, KeyEqual>::valid() const
{
    return _base.valid();
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
void
unordered_map<Key, T, Hash, KeyEqual>::clear()
{
    _base.clear();
}



template <typename Key, typename T, typename Hash, typename KeyEqual>
unordered_map<Key, T, Hash, KeyEqual>
unordered_map<Key, T, Hash, KeyEqual>::createDeviceObject(const index_t& capacity)
{
    STDGPU_EXPECTS(capacity > 0);

    unordered_map<Key, T, Hash, KeyEqual> result;
    result._base = detail::unordered_base<key_type, value_type, detail::select1st<value_type>, hasher, key_equal>::createDeviceObject(capacity);

    return result;
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
unordered_map<Key, T, Hash, KeyEqual>
unordered_map<Key, T, Hash, KeyEqual>::createDeviceObject(const index_t& bucket_count,
                                                          const index_t& excess_count)
{
    STDGPU_EXPECTS(bucket_count > 0);
    STDGPU_EXPECTS(excess_count > 0);
    STDGPU_EXPECTS(has_single_bit<std::size_t>(static_cast<std::size_t>(bucket_count)));

    unordered_map<Key, T, Hash, KeyEqual> result;
    result._base = detail::unordered_base<key_type, value_type, detail::select1st<value_type>, hasher, key_equal>::createDeviceObject(bucket_count, excess_count);

    return result;
}


template <typename Key, typename T, typename Hash, typename KeyEqual>
void
unordered_map<Key, T, Hash, KeyEqual>::destroyDeviceObject(unordered_map<Key, T, Hash, KeyEqual>& device_object)
{
    detail::unordered_base<key_type, value_type, detail::select1st<value_type>, hasher, key_equal>::destroyDeviceObject(device_object._base);
}

} // namespace stdgpu



#endif // STDGPU_UNORDERED_MAP_DETAIL_H
