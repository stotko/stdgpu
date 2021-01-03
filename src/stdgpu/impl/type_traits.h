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



namespace stdgpu
{

namespace detail
{

/**
 * \brief Macro to conditionally enable overloads
 *
 * Usage and limitations:
 *  - Must be used as last argument within a template argument list
 *  - Can be used in up to 2 function overloads with identical signature
 */
#define STDGPU_DETAIL_OVERLOAD_IF(...) typename stdgpu_DummyType = void, std::enable_if_t<__VA_ARGS__, stdgpu_DummyType>* = nullptr

/**
 * \brief Corresponding overload macro used for out-of-class member function definitions
 */
#define STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(...) typename stdgpu_DummyType, std::enable_if_t<__VA_ARGS__, stdgpu_DummyType>*

} // namespace detail

} // namespace stdgpu



#endif // STDGPU_TYPE_TRAITS_H
