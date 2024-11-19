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

#include <stdgpu/cuda/memory.h>

#include <stdgpu/cuda/impl/error.h>

namespace stdgpu::cuda
{

void
malloc_device(void** array, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMalloc(array, static_cast<std::size_t>(bytes)));
}

void
malloc_host(void** array, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMallocHost(array, static_cast<std::size_t>(bytes)));
}

void
free_device(void* array)
{
    STDGPU_CUDA_SAFE_CALL(cudaFree(array));
}

void
free_host(void* array)
{
    STDGPU_CUDA_SAFE_CALL(cudaFreeHost(array));
}

void
memcpy_device_to_device(void* destination, const void* source, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToDevice));
}

void
memcpy_device_to_host(void* destination, const void* source, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost));
}

void
memcpy_host_to_device(void* destination, const void* source, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
}

void
memcpy_host_to_host(void* destination, const void* source, index64_t bytes)
{
    STDGPU_CUDA_SAFE_CALL(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes), cudaMemcpyHostToHost));
}

} // namespace stdgpu::cuda
