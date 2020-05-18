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



namespace stdgpu
{

//! @cond Doxygen_Suppress
template <typename Key>
struct hash;
//! @endcond

/**
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
    using sfinae = std::enable_if_t<std::is_enum<E>::value, E>;
};

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/functional_detail.h>



#endif // STDGPU_FUNCTIONAL_H
