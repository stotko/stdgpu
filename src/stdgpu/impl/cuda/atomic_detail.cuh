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

#ifndef STDGPU_CUDA_ATOMIC_DETAIL_H
#define STDGPU_CUDA_ATOMIC_DETAIL_H

#include <stdgpu/contract.h>
#include <stdgpu/limits.h>
#include <stdgpu/platform.h>



inline __device__ unsigned long long int
atomicSub(unsigned long long int* address,
          const unsigned long long int value)
{
    // Slow version
    /*
    unsigned long long int old = *address, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed - value);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return old;
    */

    // Fast version
    return atomicAdd(address, stdgpu::numeric_limits<unsigned long long int>::max() - value + 1);
}


inline __device__ float
atomicSub(float* address,
          const float value)
{
    // Slow version
    /*
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( __int_as_float(assumed) - value ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return old;
    */

    // Fast version
    return atomicAdd(address, -value);
}


inline __device__ float
atomicMin(float* address,
          const float value)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( fminf(__int_as_float(assumed), value) ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __int_as_float(old);
}


inline __device__ float
atomicMax(float* address,
          const float value)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( fmaxf(__int_as_float(assumed), value) ));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __int_as_float(old);
}


inline __device__ float
atomicMinPositive(float* address,
                  const float value)
{
    STDGPU_EXPECTS(*address >= 0.0f);
    STDGPU_EXPECTS(value >= 0.0f);

    return __int_as_float(atomicMin(reinterpret_cast<int*>(address), __float_as_int(value)));
}


inline __device__ float
atomicMaxPositive(float* address,
                  const float value)
{
    STDGPU_EXPECTS(*address >= 0.0f);
    STDGPU_EXPECTS(value >= 0.0f);

    return __int_as_float(atomicMax(reinterpret_cast<int*>(address), __float_as_int(value)));
}



#endif // STDGPU_CUDA_ATOMIC_DETAIL_H
