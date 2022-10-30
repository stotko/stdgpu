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

#ifndef STDGPU_VECTOR_H
#define STDGPU_VECTOR_H

#include <stdgpu/impl/platform_check.h>

/**
 * \addtogroup vector vector
 * \ingroup data_structures
 * @{
 */

/**
 * \file stdgpu/vector.cuh
 */

#include <stdgpu/atomic.cuh>
#include <stdgpu/bitset.cuh>
#include <stdgpu/cstddef.h>
#include <stdgpu/impl/type_traits.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/mutex.cuh>
#include <stdgpu/platform.h>
#include <stdgpu/ranges.h>
#include <stdgpu/utility.h>

///////////////////////////////////////////////////////////

#include <stdgpu/vector_fwd>

///////////////////////////////////////////////////////////

namespace stdgpu
{

namespace detail
{

template <typename T, typename Allocator, typename ValueIterator, bool>
class vector_insert;

template <typename T, typename Allocator, bool>
class vector_erase;

template <typename T, typename Allocator>
void
vector_clear_iota(vector<T, Allocator>& v, const T& value);

} // namespace detail

/**
 * \ingroup vector
 * \brief A generic container similar to std::vector on the GPU
 * \tparam T The type of the stored elements
 * \tparam Allocator The allocator type
 *
 * Differences to std::vector:
 *  - index_type instead of size_type
 *  - Manual allocation and destruction of container required
 *  - max_size and capacity limited to initially allocated size
 *  - No guaranteed valid state when reaching capacity limit
 *  - Additional non-standard capacity functions full() and valid()
 *  - insert() and erase() only implemented for special case with device_end()
 *  - Some member functions missing
 */
template <typename T, typename Allocator>
class vector
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

    static_assert(!std::is_same_v<T, bool>, "std::vector<bool> specialization not provided");

    /**
     * \brief Creates an object of this class on the GPU (device)
     * \param[in] capacity The capacity of the object
     * \param[in] allocator The allocator instance to use
     * \return A newly created object of this class allocated on the GPU (device)
     * \pre capacity > 0
     */
    static vector<T, Allocator>
    createDeviceObject(const index_t& capacity, const Allocator& allocator = Allocator());

    /**
     * \brief Destroys the given object of this class on the GPU (device)
     * \param[in] device_object The object allocated on the GPU (device)
     */
    static void
    destroyDeviceObject(vector<T, Allocator>& device_object);

    /**
     * \brief Empty constructor
     */
    vector() = default;

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
     * \brief Inserts the given range of elements into the container
     * \param[in] position The position after which to insert the range
     * \param[in] begin The begin of the range
     * \param[in] end The end of the range
     * \note position must be equal to device_end()
     */
    template <typename ValueIterator, STDGPU_DETAIL_OVERLOAD_IF(detail::is_iterator_v<ValueIterator>)>
    void
    insert(device_ptr<const T> position, ValueIterator begin, ValueIterator end);

    /**
     * \brief Deletes the given range from the container
     * \param[in] begin The begin of the range
     * \param[in] end The end of the range
     * \note end must be equal to device_end()
     */
    void
    erase(device_ptr<const T> begin, device_ptr<const T> end);

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
     * \brief Creates a pointer to the begin of the device container
     * \return A pointer to the begin of the object
     */
    device_ptr<T>
    device_begin();

    /**
     * \brief Creates a pointer to the end of the device container
     * \return A pointer to the end of the object
     */
    device_ptr<T>
    device_end();

    /**
     * \brief Creates a pointer to the begin of the device container
     * \return A const pointer to the begin of the object
     */
    device_ptr<const T>
    device_begin() const;

    /**
     * \brief Creates a pointer to the end of the device container
     * \return A const pointer to the end of the object
     */
    device_ptr<const T>
    device_end() const;

    /**
     * \brief Creates a pointer to the begin of the device container
     * \return A const pointer to the begin of the object
     */
    device_ptr<const T>
    device_cbegin() const;

    /**
     * \brief Creates a pointer to the end of the device container
     * \return A const pointer to the end of the object
     */
    device_ptr<const T>
    device_cend() const;

    /**
     * \brief Creates a range of the device container
     * \return A range of the object
     */
    stdgpu::device_range<T>
    device_range();

    /**
     * \brief Creates a range of the device container
     * \return A const range of the object
     */
    stdgpu::device_range<const T>
    device_range() const;

private:
    template <typename T2, typename Allocator2, typename ValueIterator2, bool>
    friend class detail::vector_insert;

    template <typename T2, typename Allocator2, bool>
    friend class detail::vector_erase;

    friend void
    detail::vector_clear_iota<T, Allocator>(vector<T, Allocator>& v, const T& value);

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
    using atomic_allocator_type = typename stdgpu::allocator_traits<allocator_type>::template rebind_alloc<int>;

    vector(const mutex_array<mutex_default_type, mutex_array_allocator_type>& locks,
           const bitset<bitset_default_type, bitset_allocator_type>& occupied,
           const atomic<int, atomic_allocator_type>& size,
           const Allocator& allocator);

    T* _data = nullptr;
    mutex_array<mutex_default_type, mutex_array_allocator_type> _locks = {};
    bitset<bitset_default_type, bitset_allocator_type> _occupied = {};
    atomic<int, atomic_allocator_type> _size = {};
    allocator_type _allocator = {};
};

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/vector_detail.cuh>

#endif // STDGPU_VECTOR_H
