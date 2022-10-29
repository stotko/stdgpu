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

#ifndef STDGPU_DEQUE_H
#define STDGPU_DEQUE_H

#include <stdgpu/impl/platform_check.h>

/**
 * \addtogroup deque deque
 * \ingroup data_structures
 * @{
 */

/**
 * \file stdgpu/deque.cuh
 */

#include <stdgpu/atomic.cuh>
#include <stdgpu/bitset.cuh>
#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/mutex.cuh>
#include <stdgpu/platform.h>
#include <stdgpu/ranges.h>
#include <stdgpu/utility.h>

///////////////////////////////////////////////////////////

#include <stdgpu/deque_fwd>

///////////////////////////////////////////////////////////

namespace stdgpu
{

/**
 * \ingroup deque
 * \brief A generic container similar to std::deque on the GPU
 * \tparam T The type of the stored elements
 * \tparam Allocator The allocator type
 *
 * Differences to std::deque:
 *  - index_type instead of size_type
 *  - Manual allocation and destruction of container required
 *  - max_size and capacity limited to initially allocated size
 *  - No guaranteed valid state when reaching capacity limit
 *  - Additional non-standard capacity functions full(), capacity(), data(), and valid()
 *  - Some member functions missing
 *  - operator[] uses the internal begin position and may be invalidated during concurrent push_front or pop_front
 * operations
 */
template <typename T, typename Allocator>
class deque
{
public:
    using value_type = T; /**< T */

    using allocator_type = Allocator; /**< Allocator */

    using index_type = index_t;             /**< index_t */
    using difference_type = std::ptrdiff_t; /**< std::ptrdiff_t */

    using reference = value_type&;             /**< value_type& */
    using const_reference = const value_type&; /**< const value_type& */
    using pointer = value_type*;               /**< value_type* */
    using const_pointer = const value_type*;   /**< const value_type* */

    /**
     * \brief Creates an object of this class on the GPU (device)
     * \param[in] capacity The capacity of the object
     * \param[in] allocator The allocator instance to use
     * \return A newly created object of this class allocated on the GPU (device)
     * \pre capacity > 0
     */
    static deque<T, Allocator>
    createDeviceObject(const index_t& capacity, const Allocator& allocator = Allocator());

    /**
     * \brief Destroys the given object of this class on the GPU (device)
     * \param[in] device_object The object allocated on the GPU (device)
     */
    static void
    destroyDeviceObject(deque<T, Allocator>& device_object);

    /**
     * \brief Empty constructor
     */
    deque() noexcept = default;

    /**
     * \brief Returns the container allocator
     * \return The container allocator
     */
    STDGPU_HOST_DEVICE allocator_type
    get_allocator() const noexcept;

    /**
     * \brief Reads the value at position n
     * \param[in] n The position
     * \return The value at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY reference
    at(const index_type n);

    /**
     * \brief Reads the value at position n
     * \param[in] n The position
     * \return The value at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY const_reference
    at(const index_type n) const;

    /**
     * \brief Reads the value at position n
     * \param[in] n The position
     * \return The value at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY reference
    operator[](const index_type n);

    /**
     * \brief Reads the value at position n
     * \param[in] n The position
     * \return The value at this position
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY const_reference
    operator[](const index_type n) const;

    /**
     * \brief Reads the first value
     * \return The first value
     */
    STDGPU_DEVICE_ONLY reference
    front();

    /**
     * \brief Reads the first value
     * \return The first value
     */
    STDGPU_DEVICE_ONLY const_reference
    front() const;

    /**
     * \brief Reads the last value
     * \return The last value
     */
    STDGPU_DEVICE_ONLY reference
    back();

    /**
     * \brief Reads the last value
     * \return The last value
     */
    STDGPU_DEVICE_ONLY const_reference
    back() const;

    /**
     * \brief Adds the element constructed from the arguments to the end of the object
     * \param[in] args The arguments to construct the element
     * \return True if not full, false otherwise
     */
    template <class... Args>
    STDGPU_DEVICE_ONLY bool
    emplace_back(Args&&... args);

    /**
     * \brief Adds the element to the end of the object
     * \param[in] element An element
     * \return True if not full, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    push_back(const T& element);

    /**
     * \brief Removes and returns the current element from end of the object
     * \return The currently popped element and true if not empty, an empty element T() and false otherwise
     */
    STDGPU_DEVICE_ONLY pair<T, bool>
    pop_back();

    /**
     * \brief Adds the element constructed from the arguments to the front of the object
     * \param[in] args The arguments to construct the element
     * \return True if not full, false otherwise
     */
    template <class... Args>
    STDGPU_DEVICE_ONLY bool
    emplace_front(Args&&... args);

    /**
     * \brief Adds the element to the front of the object
     * \param[in] element An element
     * \return True if not full, false otherwise
     */
    STDGPU_DEVICE_ONLY bool
    push_front(const T& element);

    /**
     * \brief Removes and returns the current element from front of the object
     * \return The currently popped element and true if not empty, an empty element T() and false otherwise
     */
    STDGPU_DEVICE_ONLY pair<T, bool>
    pop_front();

    /**
     * \brief Checks if the object is empty
     * \return True if the object is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const;

    /**
     * \brief Checks if the object is full
     * \return True if the object is full, false otherwise
     */
    STDGPU_HOST_DEVICE bool
    full() const;

    /**
     * \brief Returns the current size
     * \return The size
     */
    STDGPU_HOST_DEVICE index_t
    size() const;

    /**
     * \brief Returns the maximal size
     * \return The maximal size
     */
    STDGPU_HOST_DEVICE index_t
    max_size() const noexcept;

    /**
     * \brief Returns the capacity
     * \return The capacity
     */
    STDGPU_HOST_DEVICE index_t
    capacity() const noexcept;

    /**
     * \brief Requests to shrink the capacity to the current size
     * \note This is non-binding and may not have any effect
     */
    void
    shrink_to_fit();

    /**
     * \brief Returns a pointer to the underlying data
     * \return The underlying array
     */
    const T*
    data() const noexcept;

    /**
     * \brief Returns a pointer to the underlying data
     * \return The underlying array
     */
    T*
    data() noexcept;

    /**
     * \brief Clears the complete object
     */
    void
    clear();

    /**
     * \brief Checks if the object is in a valid state
     * \return True if the state is valid, false otherwise
     */
    bool
    valid() const;

    /**
     * \brief Creates a range of the device container
     * \return A range of the object
     */
    stdgpu::device_indexed_range<T>
    device_range();

    /**
     * \brief Creates a range of the device container
     * \return A const range of the object
     */
    stdgpu::device_indexed_range<const T>
    device_range() const;

private:
    STDGPU_DEVICE_ONLY bool
    occupied(const index_t n) const;

    bool
    occupied_count_valid() const;

    bool
    size_valid() const;

    using mutex_array_allocator_type =
            typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<mutex_default_type>;
    using bitset_allocator_type =
            typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<bitset_default_type>;
    using atomic_int_allocator_type = typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<int>;
    using atomic_uint_allocator_type =
            typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<unsigned int>;
    using index_allocator_type = typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<index_t>;

    deque(const mutex_array<mutex_default_type, mutex_array_allocator_type>& locks,
          const bitset<bitset_default_type, bitset_allocator_type>& occupied,
          const atomic<int, atomic_int_allocator_type>& size,
          const atomic<unsigned int, atomic_uint_allocator_type>& begin,
          const atomic<unsigned int, atomic_uint_allocator_type>& end,
          const Allocator& allocator);

    T* _data = nullptr;
    mutable index_t* _range_indices = nullptr;
    mutex_array<mutex_default_type, mutex_array_allocator_type> _locks = {};
    bitset<bitset_default_type, bitset_allocator_type> _occupied = {};
    atomic<int, atomic_int_allocator_type> _size = {};
    atomic<unsigned int, atomic_uint_allocator_type> _begin = {};
    atomic<unsigned int, atomic_uint_allocator_type> _end = {};
    allocator_type _allocator = {};
    index_allocator_type _index_allocator = {};
};

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/deque_detail.cuh>

#endif // STDGPU_DEQUE_H
