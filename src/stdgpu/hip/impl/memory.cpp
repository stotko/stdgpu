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

#include <cstdio>

#include <stdgpu/hip/impl/error.h>

namespace stdgpu::hip
{

void
malloc(const dynamic_memory_type type, void** array, index64_t bytes)
{
    switch (type)
    {
        case dynamic_memory_type::device:
        {
            STDGPU_HIP_SAFE_CALL(hipMalloc(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::host:
        {
            STDGPU_HIP_SAFE_CALL(hipHostMalloc(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::invalid:
        default:
        {
            printf("stdgpu::hip::malloc : Unsupported dynamic memory type\n");
            return;
        }
    }
}

void
free(const dynamic_memory_type type, void* array)
{
    switch (type)
    {
        case dynamic_memory_type::device:
        {
            STDGPU_HIP_SAFE_CALL(hipFree(array));
        }
        break;

        case dynamic_memory_type::host:
        {
            STDGPU_HIP_SAFE_CALL(hipHostFree(array));
        }
        break;

        case dynamic_memory_type::invalid:
        default:
        {
            printf("stdgpu::hip::free : Unsupported dynamic memory type\n");
            return;
        }
    }
}

void
memcpy(void* destination,
       const void* source,
       index64_t bytes,
       dynamic_memory_type destination_type,
       dynamic_memory_type source_type)
{
    hipMemcpyKind kind;

    if (destination_type == dynamic_memory_type::device && source_type == dynamic_memory_type::device)
    {
        kind = hipMemcpyDeviceToDevice;
    }
    else if (destination_type == dynamic_memory_type::device && source_type == dynamic_memory_type::host)
    {
        kind = hipMemcpyHostToDevice;
    }
    else if (destination_type == dynamic_memory_type::host && source_type == dynamic_memory_type::device)
    {
        kind = hipMemcpyDeviceToHost;
    }
    else if (destination_type == dynamic_memory_type::host && source_type == dynamic_memory_type::host)
    {
        kind = hipMemcpyHostToHost;
    }
    else
    {
        printf("stdgpu::hip::memcpy : Unsupported dynamic source or destination memory type\n");
        return;
    }

    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), kind));
}

} // namespace stdgpu::hip
