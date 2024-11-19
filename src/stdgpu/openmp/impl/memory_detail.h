/*
 *  Copyright 2024 Patrick Stotko
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

#ifndef STDGPU_OPENMP_MEMORY_DETAIL_H
#define STDGPU_OPENMP_MEMORY_DETAIL_H

#include <cstring>

namespace stdgpu::openmp
{

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_device_to_device([[maybe_unused]] ExecutionPolicy&& policy,
                        void* destination,
                        const void* source,
                        index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_device_to_host([[maybe_unused]] ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_host_to_device([[maybe_unused]] ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_host_to_host([[maybe_unused]] ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

} // namespace stdgpu::openmp

#endif // STDGPU_OPENMP_MEMORY_DETAIL_H
