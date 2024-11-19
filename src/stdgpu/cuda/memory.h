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

#ifndef STDGPU_CUDA_MEMORY_H
#define STDGPU_CUDA_MEMORY_H

#include <stdgpu/cstddef.h>

namespace stdgpu::cuda
{

/**
 * \brief Performs platform-specific memory allocation on the device
 * \param[in] array A pointer to the allocated array
 * \param[in] bytes The size of the allocated array
 */
void
malloc_device(void** array, index64_t bytes);

/**
 * \brief Performs platform-specific memory allocation on the host
 * \param[in] type The type of the memory to allocate
 * \param[in] array A pointer to the allocated array
 * \param[in] bytes The size of the allocated array
 */
void
malloc_host(void** array, index64_t bytes);

/**
 * \brief Performs platform-specific memory deallocation on the device
 * \param[in] type The type of the memory to deallocate
 * \param[in] array The allocated array
 */
void
free_device(void* array);

/**
 * \brief Performs platform-specific memory deallocation on the host
 * \param[in] type The type of the memory to deallocate
 * \param[in] array The allocated array
 */
void
free_host(void* array);

/**
 * \brief Performs platform-specific memory copy from device to device
 * \param[in] destination The destination array
 * \param[in] source The source array
 * \param[in] bytes The size of the allocated array
 */
void
memcpy_device_to_device(void* destination, const void* source, index64_t bytes);

/**
 * \brief Performs platform-specific memory copy from device to host
 * \param[in] destination The destination array
 * \param[in] source The source array
 * \param[in] bytes The size of the allocated array
 */
void
memcpy_device_to_host(void* destination, const void* source, index64_t bytes);

/**
 * \brief Performs platform-specific memory copy from host to device
 * \param[in] destination The destination array
 * \param[in] source The source array
 * \param[in] bytes The size of the allocated array
 */
void
memcpy_host_to_device(void* destination, const void* source, index64_t bytes);

/**
 * \brief Performs platform-specific memory copy from host to host
 * \param[in] destination The destination array
 * \param[in] source The source array
 * \param[in] bytes The size of the allocated array
 */
void
memcpy_host_to_host(void* destination, const void* source, index64_t bytes);

} // namespace stdgpu::cuda

#endif // STDGPU_CUDA_MEMORY_H
