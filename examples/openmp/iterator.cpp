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
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/vector.cuh>        // stdgpu::vector



struct is_odd
{
    STDGPU_HOST_DEVICE bool
    operator()(const int x) const
    {
        return x % 2 == 1;
    }
};


int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // In this example, stdgpu::back_inserter is used to push all odd numbers of a sequence into a stdgpu::vector.
    //

    stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    stdgpu::vector<int> vec = stdgpu::vector<int>::createDeviceObject(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    thrust::copy_if(stdgpu::device_cbegin(d_input), stdgpu::device_cend(d_input),
                    stdgpu::back_inserter(vec),
                    is_odd());

    // vec : 1, 3, 5, ..., 99

    int sum = thrust::reduce(stdgpu::device_cbegin(vec), stdgpu::device_cend(vec),
                             0,
                             thrust::plus<int>());

    const int sum_closed_form = n * n / 4;

    std::cout << "The computed sum from i = 1 to " << n << " of i, only for odd numbers i, is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    stdgpu::vector<int>::destroyDeviceObject(vec);
}


