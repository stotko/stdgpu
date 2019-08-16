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
 * \file stdgpu/ranges.h
 */

#include <thrust/iterator/transform_iterator.h>

#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \brief A class representing a device range over an array
 * \tparam T The value type
 */
template <typename T>
class device_range
{
    public:
        using iterator      = device_ptr<T>;                    /**< device_ptr<T> */
        using value_type    = typename iterator::value_type;    /**< typename iterator::value_type */

        /**
         * \brief Constructor with automatic size inference from the given pointer
         * \param[in] p A pointer to the array
         */
        device_range(T* p);

        /**
         * \brief Constructor
         * \param[in] p A pointer to the array
         * \param[in] n The number of array elements
         */
        STDGPU_HOST_DEVICE
        device_range(T* p,
                     index_t n);

        /**
         * \brief An iterator to the begin of the range
         * \return An iterator to the begin of the range
         */
        STDGPU_HOST_DEVICE iterator
        begin();

        /**
         * \brief An iterator to the end of the range
         * \return An iterator to the end of the range
         */
        STDGPU_HOST_DEVICE iterator
        end();

    private:
        iterator _begin = {};
        iterator _end = {};
};


/**
 * \brief A class representing a host range over an array
 * \tparam T The value type
 */
template <typename T>
class host_range
{
    public:
        using iterator      = host_ptr<T>;                      /**< host_ptr<T> */
        using value_type    = typename iterator::value_type;    /**< typename iterator::value_type */

        /**
         * \brief Constructor with automatic size inference from the given pointer
         * \param[in] p A pointer to the array
         */
        host_range(T* p);

        /**
         * \brief Constructor
         * \param[in] p A pointer to the array
         * \param[in] n The number of array elements
         */
        STDGPU_HOST_DEVICE
        host_range(T* p,
                   index_t n);

        /**
         * \brief An iterator to the begin of the range
         * \return An iterator to the begin of the range
         */
        STDGPU_HOST_DEVICE iterator
        begin();

        /**
         * \brief An iterator to the end of the range
         * \return An iterator to the end of the range
         */
        STDGPU_HOST_DEVICE iterator
        end();

    private:
        iterator _begin = {};
        iterator _end = {};
};


/**
 * \brief A class representing range where a transformation is applied first
 * \tparam R The input range type
 * \tparam UnaryFunction The transformation function type
 */
template <typename R, typename UnaryFunction>
class transform_range
{
    public:
        using iterator      = thrust::transform_iterator<UnaryFunction, typename R::iterator>;      /**< thrust::transform_iterator<UnaryFunction, typename R::iterator> */
        using value_type    = typename iterator::value_type;                                        /**< typename iterator::value_type */

        /**
         * \brief Constructor
         * \param[in] r The input range
         * \param[in] f The transformation functor
         */
        STDGPU_HOST_DEVICE
        transform_range(R r,
                        UnaryFunction f);

        /**
         * \brief An iterator to the begin of the range
         * \return An iterator to the begin of the range
         */
        STDGPU_HOST_DEVICE iterator
        begin();

        /**
         * \brief An iterator to the end of the range
         * \return An iterator to the end of the range
         */
        STDGPU_HOST_DEVICE iterator
        end();

    private:
        iterator _begin = {};
        iterator _end = {};
};


namespace detail
{

/**
 * \brief A functor to map from indices to values. The constructor expects a pointer to type T.
 */
template <typename T>
struct select;

} // namespace detail


/**
 * \brief A class representing a device indexed range over a set of values
 * \tparam T The value type
 */
template <typename T>
using device_indexed_range = transform_range<device_range<index_t>, detail::select<T>>;

/**
 * \brief A class representing a host indexed range over a set of values
 * \tparam T The value type
 */
template <typename T>
using host_indexed_range = transform_range<host_range<index_t>, detail::select<T>>;

} // namespace stdgpu



#include <stdgpu/impl/ranges_detail.h>



#endif // STDGPU_RANGES_H
