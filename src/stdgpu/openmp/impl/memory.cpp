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

#include <stdgpu/openmp/memory.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>



namespace stdgpu
{
namespace openmp
{

void
dispatch_malloc(const dynamic_memory_type type,
                void** array,
                index64_t bytes)
{
    switch (type)
    {
        case dynamic_memory_type::device :
        case dynamic_memory_type::host :
        case dynamic_memory_type::managed :
        {
            *array = std::malloc(static_cast<std::size_t>(bytes)); // NOLINT(hicpp-no-malloc)
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::openmp::dispatch_malloc : Unsupported dynamic memory type\n");
            return;
        }
    }
}

void
dispatch_free(const dynamic_memory_type type,
              void* array)
{
    switch (type)
    {
        case dynamic_memory_type::device :
        case dynamic_memory_type::host :
        case dynamic_memory_type::managed :
        {
            std::free(array); // NOLINT(hicpp-no-malloc)
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::openmp::dispatch_free : Unsupported dynamic memory type\n");
            return;
        }
    }
}


void
dispatch_memcpy(void* destination,
                const void* source,
                index64_t bytes,
                dynamic_memory_type destination_type,
                dynamic_memory_type source_type)
{
    if (destination_type == dynamic_memory_type::invalid
     || source_type == dynamic_memory_type::invalid)
    {
        printf("stdgpu::openmp::dispatch_memcpy : Unsupported dynamic source or destination memory type\n");
        return;
    }

    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}


void
workaround_synchronize_device_thrust()
{
    // No synchronization workaround required for OpenMP backend
}


void
workaround_synchronize_managed_memory()
{
    // No synchronization workaround required for OpenMP backend
}


} // namespace openmp

} // namespace stdgpu

