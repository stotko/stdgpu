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

#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/unordered_set.cuh> // stdgpu::unordered_set

struct is_odd
{
    STDGPU_HOST_DEVICE bool
    operator()(const int x) const
    {
        return x % 2 == 1;
    }
};

__global__ void
insert_neighbors(const int* d_result, const stdgpu::index_t n, stdgpu::unordered_set<int> set)
{
    stdgpu::index_t i = static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n)
        return;

    int num = d_result[i];
    int num_neighborhood[3] = { num - 1, num, num + 1 };

    for (int num_neighbor : num_neighborhood)
    {
        set.insert(num_neighbor);
    }
}

int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_set is used to compute a duplicate-free set of numbers.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n / 2);
    stdgpu::unordered_set<int> set = stdgpu::unordered_set<int>::createDeviceObject(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);

    // d_input : 1, 2, 3, ..., 100

    thrust::copy_if(stdgpu::device_cbegin(d_input),
                    stdgpu::device_cend(d_input),
                    stdgpu::device_begin(d_result),
                    is_odd());

    // d_result : 1, 3, 5, ..., 99

    stdgpu::index_t threads = 32;
    stdgpu::index_t blocks = (n / 2 + threads - 1) / threads;
    insert_neighbors<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads)>>>(d_result, n / 2, set);
    cudaDeviceSynchronize();

    // set : 0, 1, 2, 3, ..., 100

    auto range_set = set.device_range();
    int sum = thrust::reduce(range_set.begin(), range_set.end(), 0, thrust::plus<int>());

    const int sum_closed_form = n * (n + 1) / 2;

    std::cout << "The duplicate-free set of numbers contains " << set.size() << " elements (" << n + 1
              << " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::unordered_set<int>::destroyDeviceObject(set);
}
