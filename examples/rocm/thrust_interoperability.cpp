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

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE


struct square
{
    STDGPU_HOST_DEVICE int
    operator()(const int x) const
    {
        return x * x;
    }
};


int
main()
{
    stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    thrust::transform(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                      stdgpu::device_begin(d_result),
                      square());

    // d_result : 1, 4, 9, ..., 10000

    int sum = thrust::reduce(stdgpu::device_begin(d_result), stdgpu::device_end(d_result),
                             0,
                             thrust::plus<int>());

    std::cout << "The computed sum from i = 1 to " << n << " of i^2 is " << sum << " (" << n * (n + 1) * (2 * n + 1) / 6 << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
}


