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

#ifndef STDGPU_FUNCTIONAL_H
#define STDGPU_FUNCTIONAL_H

/**
 * \addtogroup functional functional
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/functional.h
 */

#include <type_traits>

#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>
#include <stdgpu/utility.h>

namespace stdgpu
{

//! @cond Doxygen_Suppress
template <typename Key>
struct hash;
//! @endcond

/**
 * \ingroup functional
 * \brief A specialization for bool
 */
template <>
struct hash<bool>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const bool& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for char
 */
template <>
struct hash<char>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const char& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for singed char
 */
template <>
struct hash<signed char>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const signed char& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for unsigned char
 */
template <>
struct hash<unsigned char>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const unsigned char& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for wchar_t
 */
template <>
struct hash<wchar_t>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const wchar_t& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for char16_t
 */
template <>
struct hash<char16_t>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const char16_t& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for char32_t
 */
template <>
struct hash<char32_t>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const char32_t& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for short
 */
template <>
struct hash<short>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const short& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for unsigned short
 */
template <>
struct hash<unsigned short>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const unsigned short& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for int
 */
template <>
struct hash<int>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const int& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for unsigned int
 */
template <>
struct hash<unsigned int>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const unsigned int& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for long
 */
template <>
struct hash<long>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const long& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for unsigned long
 */
template <>
struct hash<unsigned long>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const unsigned long& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for long long
 */
template <>
struct hash<long long>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const long long& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for unsigned long long
 */
template <>
struct hash<unsigned long long>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const unsigned long long& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for float
 */
template <>
struct hash<float>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const float& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for double
 */
template <>
struct hash<double>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const double& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for long double
 */
template <>
struct hash<long double>
{
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const long double& key) const;
};

/**
 * \ingroup functional
 * \brief A specialization for all kinds of enums
 * \tparam E An enumeration
 */
template <typename E>
struct hash
{
public:
    /**
     * \brief Computes a hash value for the given key
     * \param[in] key The key
     * \return The corresponding hash value
     */
    STDGPU_HOST_DEVICE std::size_t
    operator()(const E& key) const;

private:
    /**
     * \brief Restrict specializations to enumerations
     */
    using sfinae = std::enable_if_t<std::is_enum_v<E>, E>;
};

/**
 * \ingroup functional
 * \brief A function to return the given value
 */
struct identity
{
    /**
     * \tparam T The type of the value
     * \brief Returns the given value
     * \param[in] t A value
     * \return The given value
     */
    template <typename T>
    STDGPU_HOST_DEVICE T&&
    operator()(T&& t) const noexcept;
};

/**
 * \ingroup functional
 * \brief A function to add two values
 * \tparam T The type of the values
 */
template <typename T = void>
struct plus
{
    /**
     * \brief Adds the two values
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return The sum of the given values
     */
    STDGPU_HOST_DEVICE T
    operator()(const T& lhs, const T& rhs) const;
};

/**
 * \ingroup functional
 * \brief A transparent specialization of plus
 */
template <>
struct plus<void>
{
    using is_transparent = void; /**< unspecified */

    /**
     * \brief Adds the two values
     * \tparam T The class of the first value
     * \tparam U The class of the second value
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return The sum of the given values
     */
    template <typename T, typename U>
    STDGPU_HOST_DEVICE auto
    operator()(T&& lhs, U&& rhs) const -> decltype(forward<T>(lhs) + forward<U>(rhs));
};

/**
 * \ingroup functional
 * \brief A function to perform logical AND on two values
 * \tparam T The type of the values
 */
template <typename T = void>
struct logical_and
{
    /**
     * \brief Performs logical AND on the two values
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return The result of logical AND of the given values
     */
    STDGPU_HOST_DEVICE bool
    operator()(const T& lhs, const T& rhs) const;
};

/**
 * \ingroup functional
 * \brief A transparent specialization of logical_and
 */
template <>
struct logical_and<void>
{
    using is_transparent = void; /**< unspecified */

    /**
     * \brief Performs logical AND on the two values
     * \tparam T The class of the first value
     * \tparam U The class of the second value
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return The result of logical AND of the given values
     */
    template <typename T, typename U>
    STDGPU_HOST_DEVICE auto
    operator()(T&& lhs, U&& rhs) const -> decltype(forward<T>(lhs) && forward<U>(rhs));
};

/**
 * \ingroup functional
 * \brief A function to check equality between two values
 * \tparam T The type of the values
 */
template <typename T = void>
struct equal_to
{
    /**
     * \brief Compares two values with each other
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return True if both values are equal, false otherwise
     */
    STDGPU_HOST_DEVICE bool
    operator()(const T& lhs, const T& rhs) const;
};

/**
 * \ingroup functional
 * \brief A transparent specialization of equal_to
 */
template <>
struct equal_to<void>
{
    using is_transparent = void; /**< unspecified */

    /**
     * \brief Compares two values with each other
     * \tparam T The class of the first value
     * \tparam U The class of the second value
     * \param[in] lhs The first value
     * \param[in] rhs The second value
     * \return True if both values are equal, false otherwise
     */
    template <typename T, typename U>
    STDGPU_HOST_DEVICE auto
    operator()(T&& lhs, U&& rhs) const -> decltype(forward<T>(lhs) == forward<U>(rhs));
};

/**
 * \ingroup functional
 * \brief A function to compute bitwise NOT of values
 * \tparam T The type of the values
 */
template <typename T = void>
struct bit_not
{
    /**
     * \brief Computes the bitwise NOT on the given value
     * \param[in] value The value
     * \return The result of the operation
     */
    STDGPU_HOST_DEVICE T
    operator()(const T value) const;
};

/**
 * \ingroup functional
 * \brief A transparent specialization of bit_not
 */
template <>
struct bit_not<void>
{
    using is_transparent = void; /**< unspecified */

    /**
     * \brief Computes the bitwise NOT on the given value
     * \tparam T The class of the value
     * \param[in] value The value
     * \return The result of the operation
     */
    template <typename T>
    STDGPU_HOST_DEVICE auto
    operator()(T&& value) const -> decltype(~forward<T>(value));
};

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/functional_detail.h>

#endif // STDGPU_FUNCTIONAL_H
