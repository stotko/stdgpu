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
#include <exception>
#include <hip/hip_runtime_api.h>
#include <thrust/version.h>



namespace stdgpu
{
namespace hip
{

/**
 * \brief A macro that automatically sets information about the caller
 * \param[in] error A HIP error object
 */
#define STDGPU_HIP_SAFE_CALL(error) stdgpu::hip::safe_call(error, __FILE__, __LINE__, STDGPU_FUNC)


/**
* \brief Checks whether the HIP call was successful and stops the whole program on failure
* \param[in] error An HIP error object
* \param[in] file The file from which this function was called
* \param[in] line The line from which this function was called
* \param[in] function The function from which this function was called
*/
void
safe_call(const hipError_t error,
          const char* file,
          const int line,
          const char* function)
{
    if (error != hipSuccess)
    {
        printf("stdgpu : HIP ERROR :\n"
               "  Error     : %s\n"
               "  File      : %s:%d\n"
               "  Function  : %s\n",
               hipGetErrorString(error), file, line, function);
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
            STDGPU_HIP_SAFE_CALL(hipMalloc(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::host :
        {
            STDGPU_HIP_SAFE_CALL(hipHostMalloc(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::managed :
        {
            STDGPU_HIP_SAFE_CALL(hipMallocManaged(array, static_cast<std::size_t>(bytes)));
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::hip::dispatch_malloc : Unsupported dynamic memory type\n");
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
            STDGPU_HIP_SAFE_CALL(hipFree(array));
        }
        break;

        case dynamic_memory_type::host :
        {
            STDGPU_HIP_SAFE_CALL(hipHostFree(array));
        }
        break;

        case dynamic_memory_type::managed :
        {
            STDGPU_HIP_SAFE_CALL(hipFree(array));
        }
        break;

        case dynamic_memory_type::invalid :
        default :
        {
            printf("stdgpu::hip::dispatch_free : Unsupported dynamic memory type\n");
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
    hipMemcpyKind kind;

    if ((destination_type == dynamic_memory_type::device || destination_type == dynamic_memory_type::managed)
     && (source_type == dynamic_memory_type::device || source_type == dynamic_memory_type::managed))
    {
        kind = hipMemcpyDeviceToDevice;
    }
    else if ((destination_type == dynamic_memory_type::device || destination_type == dynamic_memory_type::managed)
     && source_type == dynamic_memory_type::host)
    {
        kind = hipMemcpyHostToDevice;
    }
    else if (destination_type == dynamic_memory_type::host
     && (source_type == dynamic_memory_type::device || source_type == dynamic_memory_type::managed))
    {
        kind = hipMemcpyDeviceToHost;
    }
    else if (destination_type == dynamic_memory_type::host
     && source_type == dynamic_memory_type::host)
    {
        kind = hipMemcpyHostToHost;
    }
    else
    {
        printf("stdgpu::hip::dispatch_memcpy : Unsupported dynamic source or destination memory type\n");
        return;
    }

    STDGPU_HIP_SAFE_CALL(hipMemcpy(destination, source, static_cast<std::size_t>(bytes), kind));
}


void
workaround_synchronize_device_thrust()
{
    // We need to synchronize the device before exiting the calling function
    #if THRUST_VERSION <= 100903
        STDGPU_HIP_SAFE_CALL(hipDeviceSynchronize());
    #endif
}


void
workaround_synchronize_managed_memory()
{
    // We need to synchronize the whole device before accessing managed memory on old GPUs
    int current_device;
    int has_concurrent_managed_access;
    STDGPU_HIP_SAFE_CALL( hipGetDevice(&current_device) );
    //STDGPU_HIP_SAFE_CALL( hipDeviceGetAttribute( &hash_concurrent_managed_access, hipDevAttrConcurrentManagedAccess, current_device ) );
    has_concurrent_managed_access = 0;  // Assume that synchronization is required although the respective attribute does not exist
    if(has_concurrent_managed_access == 0)
    {
        printf("stdgpu::hip::workaround_synchronize_managed_memory : Synchronizing the whole GPU in order to access the data on the host ...\n");
        STDGPU_HIP_SAFE_CALL(hipDeviceSynchronize());
    }
}


} // namespace hip

} // namespace stdgpu

