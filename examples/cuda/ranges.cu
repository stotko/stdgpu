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

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/ranges.h>          // device_range
#include <stdgpu/unordered_set.cuh> // stdgpu::unordered_set



struct square_int
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
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates the usage of ranges with thrust at the example of stdgpu::unordered_set.
    // Furthermore, it outlines how they would be used if thrust had a proper range interface.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n);
    stdgpu::unordered_set<int> set = stdgpu::unordered_set<int>::createDeviceObject(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    auto range_int = stdgpu::device_range<int>(d_input);
    thrust::transform(range_int.begin(), range_int.end(),
                      stdgpu::device_begin(d_result),
                      square_int());

    // If thrust had a range interface (maybe in a future release), the above call could also be written in a shorter form:
    //
    // thrust::transform(stdgpu::device_range<int>(d_input),
    //                   stdgpu::device_begin(d_result),
    //                   square_int());

    // d_result : 1, 4, 9, ..., 10000

    set.insert(stdgpu::device_cbegin(d_result), stdgpu::device_cend(d_result));

    // set : 1, 4, 9, ..., 10000

    auto range_set = set.device_range();
    int sum = thrust::reduce(range_set.begin(), range_set.end(),
                             0,
                             thrust::plus<int>());

    // If thrust had a range interface (maybe in a future release), the above call could also be written in a shorter form:
    //
    // int sum = thrust::reduce(set.device_range(),
    //                          0,
    //                          thrust::plus<int>());
    //
    // Or the call to device_range may also become an implicit operation in the future.

    const int sum_closed_form = n * (n + 1) * (2 * n + 1) / 6;

    std::cout << "The computed sum from i = 1 to " << n << " of i^2 is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::unordered_set<int>::destroyDeviceObject(set);
}


