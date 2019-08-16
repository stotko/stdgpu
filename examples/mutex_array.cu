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

#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/atomic.cuh>        // stdgpu::atomic
#include <stdgpu/mutex.cuh>         // stdgpu::mutex_array
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


__global__ void
try_partial_sum(const int* d_input,
                const stdgpu::index_t n,
                stdgpu::mutex_array locks,
                int* d_result)
{
    stdgpu::index_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    stdgpu::index_t j = i % locks.size();

    // While loops might hang due to internal driver scheduling, so use a fixed number of tries
    bool finished = false;
    for (stdgpu::index_t k = 0; k < 5; ++k)
    {
        // --- SEQUENTIAL PART ---
        if (!finished && locks[j].try_lock())
        {
            // START --- critical section --- START

            d_result[j] += d_input[i];

            //  END  --- critical section ---  END
            locks[j].unlock();
            finished = true;
        }
        // --- SEQUENTIAL PART ---
    }
}


int
main()
{
    stdgpu::index_t n = 100;
    stdgpu::index_t m = 10;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(m);
    stdgpu::mutex_array locks = stdgpu::mutex_array::createDeviceObject(m);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    stdgpu::index_t threads = 128;
    stdgpu::index_t blocks = (n + threads - 1) / threads;
    try_partial_sum<<< blocks, threads >>>(d_input, n, locks, d_result);
    cudaDeviceSynchronize();

    int sum = thrust::reduce(stdgpu::device_cbegin(d_result), stdgpu::device_cend(d_result),
                             0,
                             thrust::plus<int>());

    std::cout << "The sum of all partially computed sums (via mutex locks) is " << sum << " which intentionally might not match the expected value of " << n * (n + 1) / 2 << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::mutex_array::destroyDeviceObject(locks);
}


