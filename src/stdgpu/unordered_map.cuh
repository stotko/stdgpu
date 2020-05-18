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

#ifndef STDGPU_UNORDERED_MAP_H
#define STDGPU_UNORDERED_MAP_H

/**
 * \addtogroup unordered_map unordered_map
 * \ingroup data_structures
 * @{
 */

/**
 * \file stdgpu/unordered_map.cuh
 */

#include <thrust/pair.h>

#include <stdgpu/attribute.h>
#include <stdgpu/functional.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>
#include <stdgpu/impl/unordered_base.cuh>



///////////////////////////////////////////////////////////


#include <stdgpu/unordered_map_fwd>


///////////////////////////////////////////////////////////



namespace stdgpu
{

namespace detail
{

template <typename Pair>
struct select1st;

} //namespace detail


/**
 * \brief A generic class similar to std::unordered_map on the GPU
 * \tparam Key The key type
 * \tparam T The mapped type
 * \tparam Hash The type of the hash functor
 * \tparam KeyEqual The type of the key equality functor
 *
 * Differences to std::unordered_map:
 *  - index_type instead of size_type
 *  - Manual allocation and destruction of container required
 *  - max_size and capacity limited to initially allocated size
 *  - No guaranteed valid state when reaching capacity limit
 *  - Additional non-standard capacity functions full() and valid()
 *  - Some member functions missing
 *  - Iterators may point at non-occupied and non-valid hash entry
 *  - Difference between begin() and end() returns max_size() rather than size()
 *  - Insert function returns iterator to end() rather than to the element preventing insertion
 *  - Range insert and erase functions use iterators to value_type and key_type
 */
template <typename Key,
          typename T,
          typename Hash,
          typename KeyEqual>
class unordered_map
{
    public:
        using key_type          = Key;                                      /**< Key */
        using mapped_type       = T;                                        /**< T */
        using value_type        = thrust::pair<const Key, T>;               /**< thrust::pair<const Key, T> */

        using index_type        = index_t;                                  /**< index_t */
        using difference_type   = std::ptrdiff_t;                           /**< std::ptrdiff_t */

        using key_equal         = KeyEqual;                                 /**< KeyEqual */
        using hasher            = Hash;                                     /**< Hash */

        using allocator_type    = safe_device_allocator<thrust::pair<const Key, T>>;    /**< safe_device_allocator<thrust::pair<cont Key, T>> */

        using reference         = value_type&;                              /**< value_type& */
        using const_reference   = const value_type&;                        /**< const value_type& */
        using pointer           = value_type*;                              /**< value_type* */
        using const_pointer     = const value_type*;                        /**< const value_type* */
        using iterator          = pointer;                                  /**< pointer */
        using const_iterator    = const_pointer;                            /**< const_pointer */


        /**
         * \deprecated Replaced by createDeviceObject(const index_t& capacity)
         * \brief Creates an object of this class on the GPU (device)
         * \param[in] bucket_count The number of buckets
         * \param[in] excess_count The number of excess entries
         * \pre bucket_count > 0
         * \pre excess_count > 0
         * \pre has_single_bit(bucket_count)
         * \return A newly created object of this class allocated on the GPU (device)
         */
        [[deprecated("Replaced by createDeviceObject(const index_t& capacity)")]]
        static unordered_map
        createDeviceObject(const index_t& bucket_count,
                           const index_t& excess_count);

        /**
         * \brief Creates an object of this class on the GPU (device)
         * \param[in] capacity The capacity of the object
         * \pre capacity > 0
         * \return A newly created object of this class allocated on the GPU (device)
         */
        static unordered_map
        createDeviceObject(const index_t& capacity);

        /**
         * \brief Destroys the given object of this class on the GPU (device)
         * \param[in] device_object The object allocated on the GPU (device)
         */
        static void
        destroyDeviceObject(unordered_map& device_object);


        /**
         * \brief Empty constructor
         */
        unordered_map() = default;

        /**
         * \brief Returns the container allocator
         * \return The container allocator
         */
        STDGPU_HOST_DEVICE allocator_type
        get_allocator() const;

        /**
         * \brief Checks if the object is valid
         * \return True if the state is valid, false otherwise
         */
        bool
        valid() const;


        /**
         * \brief An iterator to the begin of the internal value array
         * \return An iterator to the begin of the object
         */
        STDGPU_DEVICE_ONLY iterator
        begin();

        /**
         * \brief An iterator to the begin of the internal value array
         * \return A const iterator to the begin of the object
         */
        STDGPU_DEVICE_ONLY const_iterator
        begin() const;

        /**
         * \brief An iterator to the begin of the internal value array
         * \return A const iterator to the begin of the object
         */
        STDGPU_DEVICE_ONLY const_iterator
        cbegin() const;

        /**
         * \brief An iterator to the end of the internal value array
         * \return An iterator to the end of the object
         */
        STDGPU_DEVICE_ONLY iterator
        end();

        /**
         * \brief An iterator to the end of the internal value array
         * \return A const iterator to the end of the object
         */
        STDGPU_DEVICE_ONLY const_iterator
        end() const;

        /**
         * \brief An iterator to the end of the internal value array
         * \return A const iterator to the end of the object
         */
        STDGPU_DEVICE_ONLY const_iterator
        cend() const;


        /**
         * \brief Builds a range to the values in the container
         * \return A range of the container
         */
        device_indexed_range<const value_type>
        device_range() const;


        /**
         * \brief Returns the bucket to which the given key is mapped
         * \param[in] key The key
         * \return The bucket of the key
         * \post result < bucket_count()
         */
        STDGPU_HOST_DEVICE index_type
        bucket(const key_type& key) const;


        /**
         * \brief Returns the number of elements in the requested container bucket
         * \param[in] n The bucket index
         * \return The number of elements in the requested bucket
         */
        STDGPU_DEVICE_ONLY index_type
        bucket_size(index_type n) const;


        /**
         * \brief Returns the number of elements with the given key in the container
         * \param[in] key The key
         * \return The number of elements with the given key, i.e. 1 or 0
         */
        STDGPU_DEVICE_ONLY index_type
        count(const key_type& key) const;


        /**
         * \brief Determines if the given key is stored in the container
         * \param[in] key The key
         * \return An iterator to the position of the requested key if it was found, end() otherwise
         */
        STDGPU_DEVICE_ONLY iterator
        find(const key_type& key);

        /**
         * \brief Determines if the given key is stored in the container
         * \param[in] key The key
         * \return An iterator to the position of the requested key if it was found, end() otherwise
         */
        STDGPU_DEVICE_ONLY const_iterator
        find(const key_type& key) const;


        /**
         * \brief Determines if the given key is stored in the container
         * \param[in] key The key
         * \return True if the requested key was found, false otherwise
         */
        STDGPU_DEVICE_ONLY bool
        contains(const key_type& key) const;


        /**
         * \brief Inserts the given value into the container
         * \param[in] args The arguments to construct the element
         * \return An iterator to the inserted pair and true if the insertion was successful, end() and false otherwise
         */
        template <class... Args>
        STDGPU_DEVICE_ONLY thrust::pair<iterator, bool>
        emplace(Args&&... args);


        /**
         * \brief Inserts the given value into the container
         * \param[in] value The new value
         * \return An iterator to the inserted pair and true if the insertion was successful, end() and false otherwise
         */
        STDGPU_DEVICE_ONLY thrust::pair<iterator, bool>
        insert(const value_type& value);


        /**
         * \brief Inserts the given range of elements into the container
         * \param[in] begin The begin of the range
         * \param[in] end The end of the range
         */
        void
        insert(device_ptr<value_type> begin,
               device_ptr<value_type> end);


        /**
         * \brief Inserts the given range of elements into the container
         * \param[in] begin The begin of the range
         * \param[in] end The end of the range
         */
        void
        insert(device_ptr<const value_type> begin,
               device_ptr<const value_type> end);


        /**
         * \brief Deletes the value with the given key from the container
         * \param[in] key The key
         * \return 1 if there was a value with key and it got erased, 0 otherwise
         */
        STDGPU_DEVICE_ONLY index_type
        erase(const key_type& key);


        /**
         * \brief Deletes the values with the given range of keys from the container
         * \param[in] begin The begin of the range
         * \param[in] end The end of the range
         */
        void
        erase(device_ptr<key_type> begin,
              device_ptr<key_type> end);


        /**
         * \brief Deletes the values with the given range of keys from the container
         * \param[in] begin The begin of the range
         * \param[in] end The end of the range
         */
        void
        erase(device_ptr<const key_type> begin,
              device_ptr<const key_type> end);


        /**
         * \brief Clears the complete object
         */
        void
        clear();


        /**
         * \brief Checks if the object is empty
         * \return True if the object is empty, false otherwise
         */
        STDGPU_NODISCARD STDGPU_HOST_DEVICE bool
        empty() const;

        /**
         * \brief Checks if the object is full
         * \return True if the object is full, false otherwise
         */
        STDGPU_HOST_DEVICE bool
        full() const;

        /**
         * \brief The size
         * \return The size of the object
         */
        STDGPU_HOST_DEVICE index_t
        size() const;

        /**
         * \brief The maximum size
         * \return The maximum size
         */
        STDGPU_HOST_DEVICE index_t
        max_size() const;

        /**
         * \brief The bucket count
         * \return The number of bucket entries
         */
        STDGPU_HOST_DEVICE index_t
        bucket_count() const;

        /**
         * \deprecated Implementation detail of deprecated createDeviceObject(const index_t& bucket_count, const index_t& excess_count) function
         * \brief The excess count
         * \return The number of excess entries for handling collisions
         */
        [[deprecated("Implementation detail of deprecated createDeviceObject(const index_t& bucket_count, const index_t& excess_count) function")]]
        STDGPU_HOST_DEVICE index_t
        excess_count() const;

        /**
         * \deprecated Implementation detail of deprecated createDeviceObject(const index_t& bucket_count, const index_t& excess_count) function
         * \brief The total count
         * \return The total number of entries
         */
        [[deprecated("Implementation detail of deprecated createDeviceObject(const index_t& bucket_count, const index_t& excess_count) function")]]
        STDGPU_HOST_DEVICE index_t
        total_count() const;


        /**
         * \brief The average number of elements per bucket
         * \return The average number of elements per bucket
         */
        STDGPU_HOST_DEVICE float
        load_factor() const;

        /**
         * \brief The maximum number of elements per bucket
         * \return The maximum number of elements per bucket
         */
        STDGPU_HOST_DEVICE float
        max_load_factor() const;


        /**
         * \brief The hash function
         * \return The hash function
         */
        STDGPU_HOST_DEVICE hasher
        hash_function() const;

        /**
         * \brief The key comparator for key equality
         * \return The key comparator for key equality
         */
        STDGPU_HOST_DEVICE key_equal
        key_eq() const;

    private:
        detail::unordered_base<key_type, value_type, detail::select1st<value_type>, hasher, key_equal> _base = {};
};

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/unordered_map_detail.cuh>



#endif // STDGPU_UNORDERED_MAP_H
