/*
 *  Copyright 2020 Patrick Stotko
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

#include <stdgpu/hip/memory.h>

#include <stdgpu/hip/impl/error.h>

namespace stdgpu::hip
{

void
malloc_device(void** array, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipMalloc(array, static_cast<std::size_t>(bytes)));
}

void
malloc_host(void** array, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipHostMalloc(array, static_cast<std::size_t>(bytes)));
}

void
free_device(void* array)
{
    STDGPU_HIP_SAFE_CALL(hipFree(array));
}

void
free_host(void* array)
{
    STDGPU_HIP_SAFE_CALL(hipHostFree(array));
}

void
memcpy_device_to_device(void* destination, const void* source, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToDevice));
}

void
memcpy_device_to_host(void* destination, const void* source, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost));
}

void
memcpy_host_to_device(void* destination, const void* source, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), hipMemcpyHostToDevice));
}

void
memcpy_host_to_host(void* destination, const void* source, index64_t bytes)
{
    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), hipMemcpyHostToHost));
}

} // namespace stdgpu::hip
