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

#ifndef STDGPU_HIP_MEMORY_H
#define STDGPU_HIP_MEMORY_H

#include <stdgpu/cstddef.h>
#include <stdgpu/memory.h>

namespace stdgpu::hip
{

/**
 * \brief Performs platform-specific memory allocation
 * \param[in] type The type of the memory to allocate
 * \param[in] array A pointer to the allocated array
 * \param[in] bytes The size of the allocated array
 */
void
malloc(const dynamic_memory_type type, void** array, index64_t bytes);

/**
 * \brief Performs platform-specific memory deallocation
 * \param[in] type The type of the memory to deallocate
 * \param[in] array The allocated array
 */
void
free(const dynamic_memory_type type, void* array);

/**
 * \brief Performs platform-specific memory copy
 * \param[in] destination The destination array
 * \param[in] source The source array
 * \param[in] bytes The size of the allocated array
 * \param[in] destination_type The type of the destination array
 * \param[in] source_type The type of the source array
 */
void
memcpy(void* destination,
       const void* source,
       index64_t bytes,
       dynamic_memory_type destination_type,
       dynamic_memory_type source_type);

} // namespace stdgpu::hip

#endif // STDGPU_HIP_MEMORY_H
