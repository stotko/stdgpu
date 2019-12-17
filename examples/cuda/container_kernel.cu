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

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/unordered_set.cuh> // stdgpu::unordered_set
#include <stdgpu/vector.cuh>        // stdgpu::vector



struct is_odd
{
    STDGPU_HOST_DEVICE bool
    operator()(const int x) const
    {
        return x % 2 == 1;
    }
};


struct square
{
    STDGPU_HOST_DEVICE int
    operator()(const int x) const
    {
        return x * x;
    }
};


__global__ void
compute_neighbors_in_set(const int* d_result,
                         const stdgpu::index_t n,
                         stdgpu::unordered_set<int> set,
                         stdgpu::vector<int> vec)
{
    stdgpu::index_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int num = d_result[i];
    int num_neighborhood[2] = { num - 1, num + 1 };

    for (stdgpu::index_t j = 0; j < 2; ++j)
    {
        if (set.contains(num_neighborhood[j]))
        {
            vec.push_back(num_neighborhood[j]);
        }
    }
}


int
main()
{
    stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n);
    stdgpu::unordered_set<int> set = stdgpu::unordered_set<int>::createDeviceObject(n);
    stdgpu::vector<int> vec = stdgpu::vector<int>::createDeviceObject(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    thrust::copy_if(stdgpu::device_cbegin(d_input), stdgpu::device_cend(d_input),
                    stdgpu::inserter(set),
                    is_odd());

    // set : 1, 3, 5, ..., 99

    thrust::transform(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                      stdgpu::device_begin(d_result),
                      square());

    // d_result : 1, 4, 9, ..., 10000

    stdgpu::index_t threads = 128;
    stdgpu::index_t blocks = (n + threads - 1) / threads;
    compute_neighbors_in_set<<< blocks, threads >>>(d_result, n, set, vec);
    cudaDeviceSynchronize();

    // vec now contains the neighbors to all even square numbers smaller than 100:
    // vec: 3, 5, 15, 17, 35, 37, 63, 65, 99

    int sum = thrust::reduce(stdgpu::device_cbegin(vec), stdgpu::device_cend(vec),
                             0,
                             thrust::plus<int>());

    std::cout << "The sum of neighbors to all even square numbers smaller than 100 is " << sum << " (" << 3 + 5 + 15 + 17 + 35 + 37 + 63 + 65 + 99 << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::unordered_set<int>::destroyDeviceObject(set);
    stdgpu::vector<int>::destroyDeviceObject(vec);
}


