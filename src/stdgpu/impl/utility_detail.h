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

#ifndef STDGPU_UTILITY_DETAIL_H
#define STDGPU_UTILITY_DETAIL_H

namespace stdgpu
{

template <typename T1, typename T2>
template <
        STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_default_constructible_v<T1>&& std::is_default_constructible_v<T2>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair()
  : first()
  , second()
{
}

template <typename T1, typename T2>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_copy_constructible_v<T1>&& std::is_copy_constructible_v<T2>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(const T1& x, const T2& y)
  : first(x)
  , second(y)
{
}

template <typename T1, typename T2>
template <class U1,
          class U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(U1&& x, U2&& y)
  : first(forward<U1>(x))
  , second(forward<U2>(y))
{
}

template <typename T1, typename T2>
template <typename U1,
          typename U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_constructible_v<T1, U1&>&& std::is_constructible_v<T2, U2&>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(pair<U1, U2>& p)
  : first(p.first)
  , second(p.second)
{
}

template <typename T1, typename T2>
template <typename U1,
          typename U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(
                  std::is_constructible_v<T1, const U1&>&& std::is_constructible_v<T2, const U2&>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(const pair<U1, U2>& p)
  : first(p.first)
  , second(p.second)
{
}

template <typename T1, typename T2>
template <typename U1,
          typename U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(pair<U1, U2>&& p)
  : first(forward<U1>(p.first))
  , second(forward<U2>(p.second))
{
}

template <typename T1, typename T2>
template <typename U1,
          typename U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_constructible_v<T1, U1>&& std::is_constructible_v<T2, U2>)>
constexpr STDGPU_HOST_DEVICE
pair<T1, T2>::pair(const pair<U1, U2>&& p)
  : first(forward<const U1>(p.first))
  , second(forward<const U2>(p.second))
{
}

template <typename T1, typename T2>
template <class U1,
          class U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(
                  std::is_assignable_v<T1&, const U1&>&& std::is_assignable_v<T2&, const U2&>)>
constexpr STDGPU_HOST_DEVICE pair<T1, T2>&
pair<T1, T2>::operator=(const pair<U1, U2>& p)
{
    first = p.first;
    second = p.second;
    return *this;
}

template <typename T1, typename T2>
template <class U1,
          class U2,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_assignable_v<T1&, U1>&& std::is_assignable_v<T2&, U2>)>
constexpr STDGPU_HOST_DEVICE pair<T1, T2>&
pair<T1, T2>::operator=(pair<U1, U2>&& p)
{
    first = forward<U1>(p.first);
    second = forward<U2>(p.second);
    return *this;
}

template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>& t) noexcept
{
    return static_cast<T&&>(t);
}

template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>&& t) noexcept
{
    return static_cast<T&&>(t);
}

template <class T>
constexpr STDGPU_HOST_DEVICE std::remove_reference_t<T>&&
move(T&& t) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(t);
}

} // namespace stdgpu

#endif // STDGPU_UTILITY_DETAIL_H
