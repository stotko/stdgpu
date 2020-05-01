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

#include <cstdio>
#include <exception>
#include <thrust/version.h>
#include <cuda_runtime_api.h>   // Include after thrust to avoid redefinition warning for __host__ and __device__ in .cpp files



namespace stdgpu
{
namespace cuda
{

/**
 * \brief A macro that automatically sets information about the caller
 * \param[in] error A CUDA error object
 */
#define STDGPU_CUDA_SAFE_CALL(error) stdgpu::cuda::safe_call(error, __FILE__, __LINE__, STDGPU_FUNC)


/**
* \brief Checks whether the CUDA call was successful and stops the whole program on failure
* \param[in] error An CUDA error object
* \param[in] file The file from which this function was called
* \param[in] line The line from which this function was called
* \param[in] function The function from which this function was called
*/
void
safe_call(const cudaError_t error,
          const char* file,
          const int line,
          const char* function)
{
    if (error != cudaSuccess)
    {
        printf("stdgpu : CUDA ERROR :\n"
               "  Error     : %s\n"
               "  File      : %s:%d\n"
               "  Function  : %s\n",
               cudaGetErrorString(error), file, line, function);
        std::terminate();
    }
}


void
dispatch_malloc(const dynamic_memory_type type,
                void** array,
                index64_t bytes)
{
    switch (type)
    {
        case dynamic_memory_type::device :
        {
            STDGPU_CUDA_SAFE_CALL(cudaMalloc(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::host :
        {
            STDGPU_CUDA_SAFE_CALL(cudaMallocHost(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::managed :
        {
            STDGPU_CUDA_SAFE_CALL(cudaMallocManaged(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::cuda::dispatch_malloc : Unsupported dynamic memory type\n");
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
        {
            STDGPU_CUDA_SAFE_CALL(cudaFree(array));
        }
        break;

        case dynamic_memory_type::host :
        {
            STDGPU_CUDA_SAFE_CALL(cudaFreeHost(array));
        }
        break;

        case dynamic_memory_type::managed :
        {
            STDGPU_CUDA_SAFE_CALL(cudaFree(array));
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::cuda::dispatch_free : Unsupported dynamic memory type\n");
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
    cudaMemcpyKind kind;

    if ((destination_type == dynamic_memory_type::device || destination_type == dynamic_memory_type::managed)
     && (source_type == dynamic_memory_type::device || source_type == dynamic_memory_type::managed))
    {
        kind = cudaMemcpyDeviceToDevice;
    }
    else if ((destination_type == dynamic_memory_type::device || destination_type == dynamic_memory_type::managed)
     && source_type == dynamic_memory_type::host)
    {
        kind = cudaMemcpyHostToDevice;
    }
    else if (destination_type == dynamic_memory_type::host
     && (source_type == dynamic_memory_type::device || source_type == dynamic_memory_type::managed))
    {
        kind = cudaMemcpyDeviceToHost;
    }
    else if (destination_type == dynamic_memory_type::host
     && source_type == dynamic_memory_type::host)
    {
        kind = cudaMemcpyHostToHost;
    }
    else
    {
        printf("stdgpu::cuda::dispatch_memcpy : Unsupported dynamic source or destination memory type\n");
        return;
    }

    STDGPU_CUDA_SAFE_CALL(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes), kind));
}


void
workaround_synchronize_device_thrust()
{
    // We need to synchronize the device before exiting the calling function
    #if THRUST_VERSION <= 100903    // CUDA 10.0 and below
        STDGPU_CUDA_SAFE_CALL(cudaDeviceSynchronize());
    #endif
}


void
workaround_synchronize_managed_memory()
{
    // We need to synchronize the whole device before accessing managed memory on pre-Pascal GPUs
    int current_device;
    int hash_concurrent_managed_access;
    STDGPU_CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
    STDGPU_CUDA_SAFE_CALL( cudaDeviceGetAttribute( &hash_concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, current_device ) );
    if(hash_concurrent_managed_access == 0)
    {
        printf("stdgpu::cuda::workaround_synchronize_managed_memory : Synchronizing the whole GPU in order to access the data on the host ...\n");
        STDGPU_CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
}


} // namespace cuda

} // namespace stdgpu

