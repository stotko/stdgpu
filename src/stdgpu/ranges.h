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

#ifndef STDGPU_RANGES_H
#define STDGPU_RANGES_H

/**
 * \addtogroup ranges ranges
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/ranges.h
 */

#include <thrust/iterator/transform_iterator.h>

#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/platform.h>

namespace stdgpu
{

/**
 * \ingroup ranges
 * \brief A class representing a device range over an array
 * \tparam T The value type
 */
template <typename T>
class device_range
{
public:
    using iterator = device_ptr<T>;                             /**< device_ptr<T> */
    using value_type = typename iterator::value_type;           /**< typename iterator::value_type */
    using difference_type = typename iterator::difference_type; /**< typename iterator::difference_type */
    using reference = typename iterator::reference;             /**< typename iterator::reference */

    /**
     * \brief Empty constructor
     */
    device_range() = default;

    /**
     * \brief Constructor with automatic size inference from the given pointer
     * \param[in] p A pointer to the array
     */
    explicit device_range(T* p);

    /**
     * \brief Constructor
     * \param[in] p A pointer to the array
     * \param[in] n The number of array elements
     */
    STDGPU_HOST_DEVICE
    device_range(T* p, index64_t n);

    /**
     * \brief Constructor
     * \param[in] begin An iterator to the begin of an array
     * \param[in] n The number of array elements
     */
    STDGPU_HOST_DEVICE
    device_range(iterator begin, index64_t n);

    /**
     * \brief Constructor
     * \param[in] begin An iterator to the begin of an array
     * \param[in] end An iterator to the end of an array
     */
    STDGPU_HOST_DEVICE
    device_range(iterator begin, iterator end);

    /**
     * \brief An iterator to the begin of the range
     * \return An iterator to the begin of the range
     */
    STDGPU_HOST_DEVICE iterator
    begin() const noexcept;

    /**
     * \brief An iterator to the end of the range
     * \return An iterator to the end of the range
     */
    STDGPU_HOST_DEVICE iterator
    end() const noexcept;

    /**
     * \brief The size
     * \return The size of the range
     */
    STDGPU_HOST_DEVICE index64_t
    size() const;

    /**
     * \brief Checks if the range is empty
     * \return True if the range is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const;

private:
    iterator _begin = {};
    iterator _end = {};
};

/**
 * \ingroup ranges
 * \brief A class representing a host range over an array
 * \tparam T The value type
 */
template <typename T>
class host_range
{
public:
    using iterator = host_ptr<T>;                               /**< host_ptr<T> */
    using value_type = typename iterator::value_type;           /**< typename iterator::value_type */
    using difference_type = typename iterator::difference_type; /**< typename iterator::difference_type */
    using reference = typename iterator::reference;             /**< typename iterator::reference */

    /**
     * \brief Empty constructor
     */
    host_range() = default;

    /**
     * \brief Constructor with automatic size inference from the given pointer
     * \param[in] p A pointer to the array
     */
    explicit host_range(T* p);

    /**
     * \brief Constructor
     * \param[in] p A pointer to the array
     * \param[in] n The number of array elements
     */
    STDGPU_HOST_DEVICE
    host_range(T* p, index64_t n);

    /**
     * \brief Constructor
     * \param[in] begin An iterator to the begin of an array
     * \param[in] n The number of array elements
     */
    STDGPU_HOST_DEVICE
    host_range(iterator begin, index64_t n);

    /**
     * \brief Constructor
     * \param[in] begin An iterator to the begin of an array
     * \param[in] end An iterator to the end of an array
     */
    STDGPU_HOST_DEVICE
    host_range(iterator begin, iterator end);

    /**
     * \brief An iterator to the begin of the range
     * \return An iterator to the begin of the range
     */
    STDGPU_HOST_DEVICE iterator
    begin() const noexcept;

    /**
     * \brief An iterator to the end of the range
     * \return An iterator to the end of the range
     */
    STDGPU_HOST_DEVICE iterator
    end() const noexcept;

    /**
     * \brief The size
     * \return The size of the range
     */
    STDGPU_HOST_DEVICE index64_t
    size() const;

    /**
     * \brief Checks if the range is empty
     * \return True if the range is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const;

private:
    iterator _begin = {};
    iterator _end = {};
};

/**
 * \ingroup ranges
 * \brief A class representing range where a transformation is applied first
 * \tparam R The input range type
 * \tparam UnaryFunction The transformation function type
 */
template <typename R, typename UnaryFunction>
class transform_range
{
public:
    using iterator = thrust::transform_iterator<UnaryFunction,
                                                typename R::iterator>; /**< thrust::transform_iterator<UnaryFunction,
                                                                          typename R::iterator> */
    using value_type = typename iterator::value_type;                  /**< typename iterator::value_type */
    using difference_type = typename iterator::difference_type;        /**< typename iterator::difference_type */
    using reference = typename iterator::reference;                    /**< typename iterator::reference */

    /**
     * \brief Empty constructor
     */
    transform_range() = default;

    /**
     * \brief Constructor
     * \param[in] r The input range
     */
    STDGPU_HOST_DEVICE
    explicit transform_range(R r);

    /**
     * \brief Constructor
     * \param[in] r The input range
     * \param[in] f The transformation functor
     */
    STDGPU_HOST_DEVICE
    transform_range(R r, UnaryFunction f);

    /**
     * \brief An iterator to the begin of the range
     * \return An iterator to the begin of the range
     */
    STDGPU_HOST_DEVICE iterator
    begin() const noexcept;

    /**
     * \brief An iterator to the end of the range
     * \return An iterator to the end of the range
     */
    STDGPU_HOST_DEVICE iterator
    end() const noexcept;

    /**
     * \brief The size
     * \return The size of the range
     */
    STDGPU_HOST_DEVICE index64_t
    size() const;

    /**
     * \brief Checks if the range is empty
     * \return True if the range is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const;

private:
    iterator _begin = {};
    iterator _end = {};
};

namespace detail
{

template <typename T>
class select;

} // namespace detail

/**
 * \ingroup ranges
 * \brief A class representing a device indexed range over a set of values
 * \tparam T The value type
 */
template <typename T>
using device_indexed_range = transform_range<device_range<index_t>, detail::select<T>>;

/**
 * \ingroup ranges
 * \brief A class representing a host indexed range over a set of values
 * \tparam T The value type
 */
template <typename T>
using host_indexed_range = transform_range<host_range<index_t>, detail::select<T>>;

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/ranges_detail.h>

#endif // STDGPU_RANGES_H
