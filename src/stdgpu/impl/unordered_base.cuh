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

#ifndef STDGPU_UNORDERED_BASE_H
#define STDGPU_UNORDERED_BASE_H

#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <stdgpu/atomic.cuh>
#include <stdgpu/attribute.h>
#include <stdgpu/bitset.cuh>
#include <stdgpu/cstddef.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/mutex.cuh>
#include <stdgpu/platform.h>
#include <stdgpu/ranges.h>
#include <stdgpu/vector.cuh>



namespace stdgpu
{

namespace detail
{

/**
 * \brief The base class serving as the shared implementation of unordered_map and unordered_set
 * \tparam Key The key type
 * \tparam Value The value type
 * \tparam KeyFromValue The type of the value to key functor
 * \tparam Hash The type of the hash functor
 * \tparam KeyEqual The type of the key equality functor
 */
template <typename Key,
          typename Value,
          typename KeyFromValue,
          typename Hash,
          typename KeyEqual>
class unordered_base
{
    public:
        using key_type          = Key;                                      /**< Key */
        using value_type        = Value;                                    /**< Value */

        using index_type        = index_t;                                  /**< index_t */
        using difference_type   = std::ptrdiff_t;                           /**< std::ptrdiff_t */

        using key_from_value    = KeyFromValue;                             /**< KeyFromValue */
        using key_equal         = KeyEqual;                                 /**< KeyEqual */
        using hasher            = Hash;                                     /**< Hash */

        using allocator_type    = safe_device_allocator<Value>;             /**< safe_device_allocator<Value> */

        using reference         = value_type&;                              /**< value_type& */
        using const_reference   = const value_type&;                        /**< const value_type& */
        using pointer           = value_type*;                              /**< value_type* */
        using const_pointer     = const value_type*;                        /**< const value_type* */
        using iterator          = pointer;                                  /**< pointer */
        using const_iterator    = const_pointer;                            /**< const_pointer */


        /**
         * \brief Creates an object of this class on the GPU (device)
         * \param[in] capacity The capacity of the object
         * \pre capacity > 0
         * \return A newly created object of this class allocated on the GPU (device)
         */
        static unordered_base
        createDeviceObject(const index_t& capacity);

        /**
         * \brief Destroys the given object of this class on the GPU (device)
         * \param[in] device_object The object allocated on the GPU (device)
         */
        static void
        destroyDeviceObject(unordered_base& device_object);


        /**
         * \brief Empty constructor
         */
        unordered_base() = default;

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
         * \brief Inserts the given value into the container if possible
         * \param[in] value The new value
         * \return An iterator to the inserted pair and true if the insertion was successful, end() and false otherwise
         */
        STDGPU_DEVICE_ONLY thrust::pair<iterator, bool>
        try_insert(const value_type& value);


        /**
         * \brief Deletes any values with the given given key from the container if possible
         * \param[in] key The key
         * \return 1 if there was a value with key and it got erased, 0 otherwise
         */
        STDGPU_DEVICE_ONLY index_type
        try_erase(const key_type& key);


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
         * \note Equivalent to total_count()
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


        index_t _bucket_count = 0;                      /**< The number of buckets */                       // NOLINT(misc-non-private-member-variables-in-classes)
        index_t _excess_count = 0;                      /**< The number of excess entries */                // NOLINT(misc-non-private-member-variables-in-classes)
        value_type* _values = nullptr;                  /**< The values */                                  // NOLINT(misc-non-private-member-variables-in-classes)
        index_t* _offsets = nullptr;                    /**< The offset to model linked list */             // NOLINT(misc-non-private-member-variables-in-classes)
        bitset _occupied = {};                          /**< The indicator array for occupied entries */    // NOLINT(misc-non-private-member-variables-in-classes)
        atomic<int> _occupied_count = {};               /**< The number of occupied entries */              // NOLINT(misc-non-private-member-variables-in-classes)
        vector<index_t> _excess_list_positions = {};    /**< The excess list positions */                   // NOLINT(misc-non-private-member-variables-in-classes)
        mutex_array _locks = {};                        /**< The locks used to insert and erase entries */  // NOLINT(misc-non-private-member-variables-in-classes)
        key_from_value _key_from_value = {};            /**< The value to key functor */                    // NOLINT(misc-non-private-member-variables-in-classes)
        key_equal _key_equal = {};                      /**< The key comparison functor */                  // NOLINT(misc-non-private-member-variables-in-classes)
        hasher _hash = {};                              /**< The hashing function */                        // NOLINT(misc-non-private-member-variables-in-classes)

        mutable vector<index_t> _range_indices = {};    /**< The buffer of range indices */                 // NOLINT(misc-non-private-member-variables-in-classes)

        // Deprecated
        static unordered_base
        createDeviceObject(const index_t& bucket_count,
                           const index_t& excess_count);

        STDGPU_HOST_DEVICE index_t
        excess_count() const;

        STDGPU_HOST_DEVICE index_t
        total_count() const;

        STDGPU_DEVICE_ONLY bool
        occupied(const index_t n) const;

        STDGPU_DEVICE_ONLY index_t
        find_linked_list_end(const index_t linked_list_start);

        STDGPU_DEVICE_ONLY index_t
        find_previous_entry_position(const index_t entry_position,
                                     const index_t linked_list_start);
};

} // namespace detail

} // namespace stdgpu



#include <stdgpu/impl/unordered_base_detail.cuh>



#endif // STDGPU_UNORDERED_BASE_H
