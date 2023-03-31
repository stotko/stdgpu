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

/**
 * \file stdgpu/execution.h
 */

#include <thrust/execution_policy.h>

namespace stdgpu::execution
{

/**
 * \ingroup execution
 * \brief The device execution policy class
 */
using device_policy = std::remove_const_t<decltype(thrust::device)>;

/**
 * \ingroup execution
 * \brief The host execution policy class
 */
using host_policy = std::remove_const_t<decltype(thrust::host)>;

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
