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

#ifndef STDGPU_CUDA_BIT_DETAIL_H
#define STDGPU_CUDA_BIT_DETAIL_H



namespace stdgpu
{

namespace cuda
{

inline STDGPU_DEVICE_ONLY unsigned int
bit_width(const unsigned int number)
{
    return __ffs(number);
}


inline STDGPU_DEVICE_ONLY unsigned long long int
bit_width(const unsigned long long int number)
{
    return __ffsll(number);
}


inline STDGPU_DEVICE_ONLY int
popcount(const unsigned int number)
{
    return __popc(number);
}


inline STDGPU_DEVICE_ONLY int
popcount(const unsigned long long int number)
{
    return __popcll(number);
}

} // namespace cuda

} // namespace stdgpu



#endif // STDGPU_CUDA_BIT_DETAIL_H
