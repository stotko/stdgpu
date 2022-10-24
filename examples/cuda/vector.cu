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
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/iterator.h> // device_begin, device_end
#include <stdgpu/memory.h>   // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h> // STDGPU_HOST_DEVICE
#include <stdgpu/vector.cuh> // stdgpu::vector

__global__ void
insert_neighbors_with_duplicates(const int* d_input, const stdgpu::index_t n, stdgpu::vector<int> vec)
{
    stdgpu::index_t i = static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n)
        return;

    int num = d_input[i];
    int num_neighborhood[3] = { num - 1, num, num + 1 };

    for (int num_neighbor : num_neighborhood)
    {
        vec.push_back(num_neighbor);
    }
}

int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::vector is used to compute a set of duplicated numbers.
    // Every number is contained 3 times, except for the first and last one which is contained only 2 times.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    stdgpu::vector<int> vec = stdgpu::vector<int>::createDeviceObject(3 * n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);

    // d_input : 1, 2, 3, ..., 100

    stdgpu::index_t threads = 32;
    stdgpu::index_t blocks = (n + threads - 1) / threads;
    insert_neighbors_with_duplicates<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads)>>>(d_input,
                                                                                                                n,
                                                                                                                vec);
    cudaDeviceSynchronize();

    // vec : 0, 1, 1, 2, 2, 2, 3, 3, 3,  ..., 99, 99, 99, 100, 100, 101

    auto range_vec = vec.device_range();
    int sum = thrust::reduce(range_vec.begin(), range_vec.end(), 0, thrust::plus<int>());

    const int sum_closed_form = 3 * (n * (n + 1) / 2);

    std::cout << "The set of duplicated numbers contains " << vec.size() << " elements (" << 3 * n
              << " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    stdgpu::vector<int>::destroyDeviceObject(vec);
}
