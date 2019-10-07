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

#ifndef STDGPU_CUDA_ATOMIC_H
#define STDGPU_CUDA_ATOMIC_H



/**
 * \brief Atomically computes the difference of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ unsigned long long int
atomicSub(unsigned long long int* address,
          const unsigned long long int value);

/**
 * \brief Atomically computes the difference of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicSub(float* address,
          const float value);


/**
 * \brief Atomically computes the minimum of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicMin(float* address,
          const float value);


/**
 * \brief Atomically computes the maximum of the two values
 * \param[in] address A pointer to a value
 * \param[in] value A value
 * \return The old value at the given address
 */
__device__ float
atomicMax(float* address,
          const float value);


/**
 * \brief Atomically computes the minimum of the two positive values
 * \param[in] address A pointer to a positive value
 * \param[in] value A positive value
 * \return The old value at the given address
 */
__device__ float
atomicMinPositive(float* address,
                  const float value);


/**
 * \brief Atomically computes the maximum of the two positive values
 * \param[in] address A pointer to a positive value
 * \param[in] value A positive value
 * \return The old value at the given address
 */
__device__ float
atomicMaxPositive(float* address,
                  const float value);



#include <stdgpu/impl/cuda/atomic_detail.cuh>



#endif // STDGPU_CUDA_ATOMIC_H
