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

#ifndef STDGPU_HIP_MEMORY_DETAIL_H
#define STDGPU_HIP_MEMORY_DETAIL_H

#include <thrust/detail/execution_policy.h>
#include <thrust/system/hip/detail/util.h>

#include <stdgpu/hip/impl/error.h>

namespace stdgpu::hip
{

template <typename ExecutionPolicy, STDGPU_DETAIL_OVERLOAD_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_impl(ExecutionPolicy&& policy,
            void* destination,
            const void* source,
            index64_t bytes,
            hipMemcpyKind kind,
            bool needs_sychronization)
{
    cudaStream_t stream =
            thrust::hip_rocprim::stream(thrust::detail::derived_cast(thrust::detail::strip_const(policy)));

    STDGPU_HIP_SAFE_CALL(hipMemcpyAsync(destination, source, static_cast<std::size_t>(bytes), kind, stream));
    if (needs_sychronization)
    {
        STDGPU_HIP_SAFE_CALL(hipStreamSynchronize(stream));
    }
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_device_to_device(ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    memcpy_impl(std::forward<ExecutionPolicy>(policy), destination, source, bytes, hipMemcpyDeviceToDevice, false);
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_device_to_host(ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    memcpy_impl(std::forward<ExecutionPolicy>(policy), destination, source, bytes, hipMemcpyDeviceToHost, true);
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_host_to_device(ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    memcpy_impl(std::forward<ExecutionPolicy>(policy), destination, source, bytes, hipMemcpyHostToDevice, false);
}

template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
memcpy_host_to_host(ExecutionPolicy&& policy, void* destination, const void* source, index64_t bytes)
{
    memcpy_impl(std::forward<ExecutionPolicy>(policy), destination, source, bytes, hipMemcpyHostToHost, true);
}

} // namespace stdgpu::hip

#endif // STDGPU_HIP_MEMORY_DETAIL_H
