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

#include <cstdlib>
#include <cstring>

namespace stdgpu::openmp
{

void
malloc_device(void** array, index64_t bytes)
{
    *array = std::malloc(static_cast<std::size_t>(bytes)); // NOLINT(hicpp-no-malloc,cppcoreguidelines-no-malloc)
}

void
malloc_host(void** array, index64_t bytes)
{
    *array = std::malloc(static_cast<std::size_t>(bytes)); // NOLINT(hicpp-no-malloc,cppcoreguidelines-no-malloc)
}

void
free_device(void* array)
{
    std::free(array); // NOLINT(hicpp-no-malloc,cppcoreguidelines-no-malloc)
}

void
free_host(void* array)
{
    std::free(array); // NOLINT(hicpp-no-malloc,cppcoreguidelines-no-malloc)
}

void
memcpy_device_to_device(void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

void
memcpy_device_to_host(void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

void
memcpy_host_to_device(void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

void
memcpy_host_to_host(void* destination, const void* source, index64_t bytes)
{
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
}

} // namespace stdgpu::openmp
