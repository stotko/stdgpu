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

#ifndef STDGPU_UTILITY_H
#define STDGPU_UTILITY_H

/**
 * \defgroup utility utility
 * \ingroup utilities
 */

/**
 * \file stdgpu/utility.h
 */

#include <type_traits>

#include <stdgpu/platform.h>
#include <stdgpu/type_traits.h>

namespace stdgpu
{

/**
 * \ingroup utility
 * \tparam T1 The type of the first value
 * \tparam T2 The type of the second value
 * \brief A pair of two values of potentially different types
 */
template <typename T1, typename T2>
struct pair
{
    using first_type = T1;  /**< T1 */
    using second_type = T2; /**< T2 */

    /**
     * \brief Default Destructor
     */
    ~pair() = default;

    /**
     * \brief Constructor
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_default_constructible_v<T1>&& std::is_default_constructible_v<T2>)>
    constexpr STDGPU_HOST_DEVICE
    pair();

    /**
     * \brief Constructor
     * \param[in] x The new first element
     * \param[in] y The new second element
     */
    template <STDGPU_DETAIL_OVERLOAD_IF(std::is_copy_constructible_v<T1>&& std::is_copy_constructible_v<T2>)>
    constexpr STDGPU_HOST_DEVICE
    pair(const T1& x, const T2& y);

    /**
     * \brief Constructor
     * \tparam U1 The type of the first element
     * \tparam U2 The type of the second element
     * \param[in] x The new first element
     * \param[in] y The new second element
     */
    template <class U1 = T1,
              class U2 = T2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
    constexpr STDGPU_HOST_DEVICE
    pair(U1&& x, U2&& y);

    /**
     * \brief Copy constructor
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The pair to copy from
     */
    template <typename U1,
              typename U2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_constructible_v<T1, U1&>&& std::is_constructible_v<T2, U2&>)>
    constexpr STDGPU_HOST_DEVICE
    pair(pair<U1, U2>& p); // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Copy constructor
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The copied pair
     */
    template <
            typename U1,
            typename U2,
            STDGPU_DETAIL_OVERLOAD_IF(std::is_constructible_v<T1, const U1&>&& std::is_constructible_v<T2, const U2&>)>
    constexpr STDGPU_HOST_DEVICE
    pair(const pair<U1, U2>& p); // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Move constructor
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The moved pair
     */
    template <class U1,
              class U2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
    constexpr STDGPU_HOST_DEVICE
    pair(pair<U1, U2>&& p); // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Move constructor
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The moved pair
     */
    template <class U1,
              class U2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
    constexpr STDGPU_HOST_DEVICE
    pair(const pair<U1, U2>&& p); // NOLINT(hicpp-explicit-conversions)

    /**
     * \brief Default copy constructor
     * \param[in] p The copied pair
     */
    pair(const pair& p) = default;

    /**
     * \brief Default move constructor
     * \param[in] p The moved pair
     */
    pair(pair&& p) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)

    /**
     * \brief Default copy assignment operator
     * \param[in] p The pair to copy from
     * \return *this
     */
    pair&
    operator=(const pair& p) = default;

    /**
     * \brief Copy assignment operator
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The pair to copy from
     * \return *this
     */
    template <class U1,
              class U2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_assignable_v<T1&, const U1&>&& std::is_assignable_v<T2&, const U2&>)>
    constexpr STDGPU_HOST_DEVICE pair&
    operator=(const pair<U1, U2>& p);

    /**
     * \brief Default move assignment operator
     * \param[in] p The moved pair
     * \return *this
     */
    pair&
    operator=(pair&& p) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)

    /**
     * \brief Move assignment operator
     * \tparam U1 The type of the other pair's first element
     * \tparam U2 The type of the other pair's second element
     * \param[in] p The moved pair
     * \return *this
     */
    template <class U1,
              class U2,
              STDGPU_DETAIL_OVERLOAD_IF(std::is_assignable_v<T1&, U1>&& std::is_assignable_v<T2&, U2>)>
    constexpr STDGPU_HOST_DEVICE pair&
    operator=(pair<U1, U2>&& p);

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members,misc-non-private-member-variables-in-classes)
    first_type first; /**< First element of pair */
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members,misc-non-private-member-variables-in-classes)
    second_type second; /**< Second element of pair */
};

/**
 * \ingroup utility
 * \brief Forwards a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The forwarded value
 */
template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>& t) noexcept;

/**
 * \ingroup utility
 * \brief Forwards a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The forwarded value
 */
template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>&& t) noexcept;

/**
 * \ingroup utility
 * \brief Moves a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The moved value
 */
template <class T>
constexpr STDGPU_HOST_DEVICE std::remove_reference_t<T>&&
move(T&& t) noexcept;

} // namespace stdgpu

#include <stdgpu/impl/utility_detail.h>

#endif // STDGPU_UTILITY_H
