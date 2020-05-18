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

#ifndef STDGPU_ITERATOR_H
#define STDGPU_ITERATOR_H

/**
 * \addtogroup iterator iterator
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/iterator.h
 */

#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h> // Only forward declaration included by thrust/detail/pointer.h

#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \ingroup iterator
 * \brief A host pointer class allowing to call thrust algorithms without explicitly using the thrust::device execution policy
 * \note Considered equivalent to thrust::device_ptr but can be processed in plain C++
 */
template <typename T>
using device_ptr = thrust::pointer<T, thrust::device_system_tag>;

/**
 * \ingroup iterator
 * \brief A host pointer class allowing to call thrust algorithms without explicitly using the thrust::host execution policy
 */
template <typename T>
using host_ptr = thrust::pointer<T, thrust::host_system_tag>;

} // namespace stdgpu


//! @cond Doxygen_Suppress
namespace std
{

template <typename T>
struct iterator_traits<stdgpu::device_ptr<T>>
{
    using difference_type   = typename std::iterator_traits<T*>::difference_type;
    using value_type        = typename std::iterator_traits<T*>::value_type;
    using pointer           = typename std::iterator_traits<T*>::pointer;
    using reference         = typename std::iterator_traits<T*>::reference;
    using iterator_category = typename stdgpu::device_ptr<T>::iterator_category;
};

template <typename T>
struct iterator_traits<stdgpu::host_ptr<T>>
{
    using difference_type   = typename std::iterator_traits<T*>::difference_type;
    using value_type        = typename std::iterator_traits<T*>::value_type;
    using pointer           = typename std::iterator_traits<T*>::pointer;
    using reference         = typename std::iterator_traits<T*>::reference;
    using iterator_category = typename stdgpu::host_ptr<T>::iterator_category;
};

} // namespace std
//! @endcond


namespace stdgpu
{

/**
 * \ingroup iterator
 * \brief Constructs a device_ptr object
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A device_ptr pointing to the array
 */
template <typename T>
STDGPU_HOST_DEVICE device_ptr<T>
make_device(T* device_array);


/**
 * \ingroup iterator
 * \brief Constructs a host_ptr object
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A host_ptr pointing to the array
 */
template <typename T>
STDGPU_HOST_DEVICE host_ptr<T>
make_host(T* host_array);


/**
 * \ingroup iterator
 * \brief Finds the size (number of elements) of the given dynamically allocated array
 * \tparam T The type of the array
 * \param[in] array An array
 * \return The size (number of elements) of the given array
 * \note The array must be created by this API, otherwise an assertion is triggered
 */
template <typename T>
index64_t
size(T* array);


/**
 * \ingroup iterator
 * \brief Specialization for unknown data type: Finds the size (in bytes) of the given dynamically allocated array
 * \param[in] array An array
 * \return The size (in bytes) of the given array
 * \note The array must be created by this API, otherwise an assertion is triggered
 */
template <>
index64_t
size(void* array);



/**
 * \ingroup iterator
 * \brief Creates a pointer to the begin of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A pointer to the begin of the host array
 */
template <typename T>
host_ptr<T>
host_begin(T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a pointer to the end of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A pointer to the end of the host array
 */
template <typename T>
host_ptr<T>
host_end(T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a pointer to the begin of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A pointer to the begin of the device array
 */
template <typename T>
device_ptr<T>
device_begin(T* device_array);


/**
 * \ingroup iterator
 * \brief Creates a pointer to the end of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A pointer to the end of the device array
 */
template <typename T>
device_ptr<T>
device_end(T* device_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the begin of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A constant pointer to the begin of the host array
 */
template <typename T>
host_ptr<const T>
host_begin(const T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the end of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A constant pointer to the end of the host array
 */
template <typename T>
host_ptr<const T>
host_end(const T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the begin of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A constant pointer to the begin of the device array
 */
template <typename T>
device_ptr<const T>
device_begin(const T* device_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the end of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A constant pointer to the end of the device array
 */
template <typename T>
device_ptr<const T>
device_end(const T* device_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the begin of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A constant pointer to the begin of the host array
 */
template <typename T>
host_ptr<const T>
host_cbegin(const T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the end of the given host array
 * \tparam T The type of the array
 * \param[in] host_array An array
 * \return A constant pointer to the end of the host array
 */
template <typename T>
host_ptr<const T>
host_cend(const T* host_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the begin of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A constant pointer to the begin of the device array
 */
template <typename T>
device_ptr<const T>
device_cbegin(const T* device_array);


/**
 * \ingroup iterator
 * \brief Creates a constant pointer to the end of the given device array
 * \tparam T The type of the array
 * \param[in] device_array An array
 * \return A constant pointer to the end of the device array
 */
template <typename T>
device_ptr<const T>
device_cend(const T* device_array);



/**
 * \ingroup iterator
 * \brief Creates a pointer to the begin of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A pointer to the begin of the host container
 */
template <typename C>
auto
host_begin(C& host_container) -> decltype(host_container.host_begin());


/**
 * \ingroup iterator
 * \brief Creates a pointer to the end of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A pointer to the end of the host container
 */
template <typename C>
auto
host_end(C& host_container) -> decltype(host_container.host_end());


/**
 * \ingroup iterator
 * \brief Creates a pointer to the begin of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A pointer to the begin of the device container
 */
template <typename C>
auto
device_begin(C& device_container) -> decltype(device_container.device_begin());


/**
 * \ingroup iterator
 * \brief Creates a pointer to the end of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A pointer to the end of the device container
 */
template <typename C>
auto
device_end(C& device_container) -> decltype(device_container.device_end());


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the begin of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A constant pointer to the begin of the host container
 */
template <typename C>
auto
host_begin(const C& host_container) -> decltype(host_container.host_begin());


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the end of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A constant pointer to the end of the host container
 */
template <typename C>
auto
host_end(const C& host_container) -> decltype(host_container.host_end());


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the begin of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A constant pointer to the begin of the device container
 */
template <typename C>
auto
device_begin(const C& device_container) -> decltype(device_container.device_begin());


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the end of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A constant pointer to the end of the device container
 */
template <typename C>
auto
device_end(const C& device_container) -> decltype(device_container.device_end());


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the begin of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A constant pointer to the begin of the host container
 */
template <typename C>
auto
host_cbegin(const C& host_container) -> decltype(host_begin(host_container));


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the end of the given host container
 * \tparam C The type of the container
 * \param[in] host_container An array
 * \return A constant pointer to the end of the host container
 */
template <typename C>
auto
host_cend(const C& host_container) -> decltype(host_end(host_container));


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the begin of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A constant pointer to the begin of the device container
 */
template <typename C>
auto
device_cbegin(const C& device_container) -> decltype(device_begin(device_container));


/**
 * \ingroup iterator
 * \brief Creates a contant pointer to the end of the given device container
 * \tparam C The type of the container
 * \param[in] device_container An array
 * \return A constant pointer to the end of the device container
 */
template <typename C>
auto
device_cend(const C& device_container) -> decltype(device_end(device_container));


namespace detail
{

template <typename Container>
struct back_insert_iterator_base;

template <typename Container>
struct front_insert_iterator_base;

template <typename Container>
struct insert_iterator_base;

} // namespace detail


/**
 * \brief An output iterator which inserts elements into a container using push_back
 * \tparam Container The type of the container
 */
template <typename Container>
class back_insert_iterator
    : public detail::back_insert_iterator_base<Container>::type
{
    public:
        using container_type = Container;       /**< Container */

        //! @cond Doxygen_Suppress
        using super_t = typename detail::back_insert_iterator_base<Container>::type;

        friend class thrust::iterator_core_access;
        //! @endcond

        /**
         * \brief Constructor
         * \param[in] c The container into which the elements are inserted
         */
        STDGPU_HOST_DEVICE
        explicit back_insert_iterator(Container& c);

    private:
        STDGPU_HOST_DEVICE typename super_t::reference
        dereference() const;

        Container _c;
};

/**
 * \ingroup iterator
 * \brief Constructs a back_insert_iterator
 * \param[in] c The container into which the elements are inserted
 * \return A back_insert_iterator for the given container
 */
template <typename Container>
STDGPU_HOST_DEVICE back_insert_iterator<Container>
back_inserter(Container& c);


/**
 * \brief An output iterator which inserts elements into a container using push_front
 * \tparam Container The type of the container
 */
template <typename Container>
class front_insert_iterator
    : public detail::front_insert_iterator_base<Container>::type
{
    public:
        using container_type = Container;       /**< Container */

        //! @cond Doxygen_Suppress
        using super_t = typename detail::front_insert_iterator_base<Container>::type;

        friend class thrust::iterator_core_access;
        //! @endcond

        /**
         * \brief Constructor
         * \param[in] c The container into which the elements are inserted
         */
        STDGPU_HOST_DEVICE
        explicit front_insert_iterator(Container& c);

    private:
        STDGPU_HOST_DEVICE typename super_t::reference
        dereference() const;

        Container _c;
};

/**
 * \ingroup iterator
 * \brief Constructs a front_insert_iterator
 * \param[in] c The container into which the elements are inserted
 * \return A front_insert_iterator for the given container
 */
template <typename Container>
STDGPU_HOST_DEVICE front_insert_iterator<Container>
front_inserter(Container& c);


/**
 * \brief An output iterator which inserts elements into a container using insert
 * \tparam Container The type of the container
 *
 * Differences to std::insert_iterator:
 *  - Constructor without iterator position
 */
template <typename Container>
class insert_iterator
    : public detail::insert_iterator_base<Container>::type
{
    public:
        using container_type = Container;       /**< Container */

        //! @cond Doxygen_Suppress
        using super_t = typename detail::insert_iterator_base<Container>::type;

        friend class thrust::iterator_core_access;
        //! @endcond

        /**
         * \brief Constructor
         * \param[in] c The container into which the elements are inserted
         */
        STDGPU_HOST_DEVICE
        explicit insert_iterator(Container& c);

    private:
        STDGPU_HOST_DEVICE typename super_t::reference
        dereference() const;

        Container _c;
};

/**
 * \ingroup iterator
 * \brief Constructs an insert_iterator
 * \param[in] c The container into which the elements are inserted
 * \return An insert_iterator for the given container
 */
template <typename Container>
STDGPU_HOST_DEVICE insert_iterator<Container>
inserter(Container& c);

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/iterator_detail.h>



#endif // STDGPU_ITERATOR_H
