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

#ifndef STDGPU_CUDA_BIT_H
#define STDGPU_CUDA_BIT_H



namespace stdgpu
{

namespace cuda
{

/**
 * \brief Computes the smallest number of bits to represent the given number
 * \param[in] number A number
 * \return The smallest number of bits to represent the given number
 */
STDGPU_DEVICE_ONLY unsigned int
bit_width(const unsigned int number);


/**
 * \brief Computes the smallest number of bits to represent the given number
 * \param[in] number A number
 * \return The smallest number of bits to represent the given number
 */
STDGPU_DEVICE_ONLY unsigned long long int
bit_width(const unsigned long long int number);


/**
 * \brief Counts the number of set bits in the number
 * \param[in] number A number
 * \return The number of set bits
 */
STDGPU_DEVICE_ONLY int
popcount(const unsigned int number);


/**
 * \brief Counts the number of set bits in the number
 * \param[in] number A number
 * \return The number of set bits
 */
STDGPU_DEVICE_ONLY int
popcount(const unsigned long long int number);

} // namespace cuda

} // namespace stdgpu



#include <stdgpu/cuda/impl/bit_detail.cuh>


#endif // STDGPU_CUDA_BIT_H
