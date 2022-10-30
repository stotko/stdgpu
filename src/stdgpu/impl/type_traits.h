/*
 *  Copyright 2021 Patrick Stotko
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

#ifndef STDGPU_TYPE_TRAITS_H
#define STDGPU_TYPE_TRAITS_H

#include <type_traits>
#include <utility>

#include <stdgpu/impl/preprocessor.h>

namespace stdgpu::detail
{

/**
 * \brief Macro to conditionally enable overloads
 *
 * Usage and limitations:
 *  - Must be used as last argument within a template argument list
 *  - Can be used in up to 2 function overloads with identical signature
 */
#define STDGPU_DETAIL_OVERLOAD_IF(...)                                                                                 \
    typename stdgpu_DummyType = void, std::enable_if_t<__VA_ARGS__, stdgpu_DummyType>* = nullptr

/**
 * \brief Corresponding overload macro used for out-of-class member function definitions
 */
#define STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(...)                                                                      \
    typename stdgpu_DummyType, std::enable_if_t<__VA_ARGS__, stdgpu_DummyType>*

#define STDGPU_DETAIL_DEFINE_TRAIT(name, ...)                                                                          \
    template <typename T, typename = void>                                                                             \
    struct name : std::false_type                                                                                      \
    {                                                                                                                  \
    };                                                                                                                 \
                                                                                                                       \
    template <typename T>                                                                                              \
    struct name<T, std::void_t<__VA_ARGS__>> : std::true_type                                                          \
    {                                                                                                                  \
    };                                                                                                                 \
                                                                                                                       \
    template <typename T>                                                                                              \
    inline constexpr bool STDGPU_DETAIL_CAT2(name, _v) = name<T>::value;

// Clang does not detect T::pointer for thrust::device_pointer, so avoid checking it
STDGPU_DETAIL_DEFINE_TRAIT(is_iterator,
                           typename T::difference_type,
                           typename T::value_type,
                           /*typename T::pointer,*/ typename T::reference,
                           typename T::iterator_category)

STDGPU_DETAIL_DEFINE_TRAIT(is_transparent, typename T::is_transparent)

STDGPU_DETAIL_DEFINE_TRAIT(is_base, typename T::is_base)

STDGPU_DETAIL_DEFINE_TRAIT(has_get, decltype(std::declval<T>().get()))
STDGPU_DETAIL_DEFINE_TRAIT(has_arrow_operator,
                           decltype(std::declval<T>()
                                            .
                                            operator->()))

template <typename T>
struct dependent_false : std::false_type
{
};

template <typename T>
inline constexpr bool dependent_false_v = dependent_false<T>::value;

} // namespace stdgpu::detail

#endif // STDGPU_TYPE_TRAITS_H
