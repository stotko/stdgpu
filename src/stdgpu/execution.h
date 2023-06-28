/*
 *  Copyright 2022 Patrick Stotko
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

#ifndef STDGPU_EXECUTION_H
#define STDGPU_EXECUTION_H

/**
 * \defgroup execution execution
 * \ingroup utilities
 */

#include <type_traits>

#include <stdgpu/type_traits.h>

/**
 * \file stdgpu/execution.h
 */

#include <thrust/execution_policy.h>

namespace stdgpu
{

/**
 * \ingroup execution
 * \brief Type trait to check whether the provided class is an execution policy
 */
template <typename T>
struct is_execution_policy : std::is_base_of<thrust::execution_policy<T>, T>
{
};

//! @cond Doxygen_Suppress
template <typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
//! @endcond

} // namespace stdgpu

namespace stdgpu::execution
{

/**
 * \ingroup execution
 * \brief The base execution policy class from which all policies are derived and custom ones must be derived
 */
template <typename T>
using execution_policy = thrust::execution_policy<T>;

/**
 * \ingroup execution
 * \brief The device execution policy class
 */
using device_policy = std::remove_const_t<decltype(thrust::device)>;

static_assert(is_execution_policy_v<remove_cvref_t<device_policy>>,
              "stdgpu::execution::device_policy: Should be an execution policy but is not");

/**
 * \ingroup execution
 * \brief The host execution policy class
 */
using host_policy = std::remove_const_t<decltype(thrust::host)>;

static_assert(is_execution_policy_v<remove_cvref_t<host_policy>>,
              "stdgpu::execution::host_policy: Should be an execution policy but is not");

/**
 * \ingroup execution
 * \brief The device execution policy
 */
constexpr device_policy device;

/**
 * \ingroup execution
 * \brief The host execution policy
 */
constexpr host_policy host;

} // namespace stdgpu::execution

#endif // STDGPU_EXECUTION_H
