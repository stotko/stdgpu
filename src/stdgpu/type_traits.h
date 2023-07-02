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

/**
 * \defgroup type_traits type_traits
 * \ingroup utilities
 */

/**
 * \file stdgpu/type_traits.h
 */

#include <type_traits>

namespace stdgpu
{

/**
 * \ingroup type_traits
 * \brief Type trait to remove const, volative, and reference qualifiers from the given type
 * \tparam T The input type
 */
template <typename T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>; /**< type */
};

//! @cond Doxygen_Suppress
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
//! @endcond

} // namespace stdgpu

#include <stdgpu/impl/type_traits_detail.h>

#endif // STDGPU_TYPE_TRAITS_H
