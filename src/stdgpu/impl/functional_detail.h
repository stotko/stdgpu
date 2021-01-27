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

#ifndef STDGPU_FUNCTIONAL_DETAIL_H
#define STDGPU_FUNCTIONAL_DETAIL_H

#include <type_traits>

#include <stdgpu/cstddef.h>
#include <stdgpu/bit.h>



namespace stdgpu
{

#define STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(T) \
inline STDGPU_HOST_DEVICE std::size_t \
hash<T>::operator()(const T& key) const \
{ \
    return static_cast<std::size_t>(key); \
}

STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(bool)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(char)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(signed char)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(unsigned char)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(wchar_t)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(char16_t)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(char32_t)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(short)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(unsigned short)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(int)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(unsigned int)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(long)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(unsigned long)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(long long)
STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE(unsigned long long)

#undef STDGPU_DETAIL_COMPOUND_HASH_BASIC_INTEGER_TYPE

inline STDGPU_HOST_DEVICE std::size_t
hash<float>::operator()(const float& key) const
{
    return hash<std::uint32_t>()(bit_cast<std::uint32_t>(key));
}

inline STDGPU_HOST_DEVICE std::size_t
hash<double>::operator()(const double& key) const
{
    return hash<std::uint64_t>()(bit_cast<std::uint64_t>(key));
}

inline STDGPU_HOST_DEVICE std::size_t
hash<long double>::operator()(const long double& key) const
{
    return hash<double>()(static_cast<double>(key));
}


template <typename E>
inline STDGPU_HOST_DEVICE std::size_t
hash<E>::operator()(const E& key) const
{
    return hash<std::underlying_type_t<E>>()(static_cast<std::underlying_type_t<E>>(key));
}


template <typename T>
inline STDGPU_HOST_DEVICE bool
equal_to<T>::operator()(const T &lhs,
                        const T &rhs) const
{
    return lhs == rhs;
}

template <typename T, typename U>
inline STDGPU_HOST_DEVICE auto
equal_to<void>::operator()(T&& lhs,
                           U&& rhs) const -> decltype(forward<T>(lhs) == forward<U>(rhs))
{
    return forward<T>(lhs) == forward<U>(rhs);
}


template <typename T>
inline STDGPU_HOST_DEVICE T
bit_not<T>::operator()(const T value) const
{
    return ~value;
}

template <typename T>
inline STDGPU_HOST_DEVICE auto
bit_not<void>::operator()(T&& value) const -> decltype(~forward<T>(value))
{
    return ~forward<T>(value);
}

} // namespace stdgpu



#endif // STDGPU_FUNCTIONAL_DETAIL_H
