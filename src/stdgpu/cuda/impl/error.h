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

#ifndef STDGPU_CUDA_ERROR_H
#define STDGPU_CUDA_ERROR_H

#include <cuda_runtime_api.h>
#include <exception>

#include <stdgpu/cstddef.h>

namespace stdgpu::cuda
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
inline void
safe_call(const cudaError_t error, const char* file, const int line, const char* function)
{
    if (error != cudaSuccess)
    {
        printf("stdgpu : CUDA ERROR :\n"
               "  Error     : %s\n"
               "  File      : %s:%d\n"
               "  Function  : %s\n",
               cudaGetErrorString(error),
               file,
               line,
               function);
        std::terminate();
    }
}
} // namespace stdgpu::cuda

#endif // STDGPU_CUDA_ERROR_H
