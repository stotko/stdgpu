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
#include <thrust/sequence.h>

#include <stdgpu/atomic.cuh> // stdgpu::atomic
#include <stdgpu/bitset.cuh> // stdgpu::bitset
#include <stdgpu/iterator.h> // device_begin, device_end
#include <stdgpu/memory.h>   // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h> // STDGPU_HOST_DEVICE

struct is_odd
{
    STDGPU_HOST_DEVICE bool
    operator()(const int x) const
    {
        return x % 2 == 1;
    }
};

void
set_bits(const int* d_result, const stdgpu::index_t d_result_size, stdgpu::bitset<>& bits, stdgpu::atomic<int>& counter)
{
#pragma omp parallel for
    for (stdgpu::index_t i = 0; i < d_result_size; ++i)
    {
        bool was_set = bits.set(d_result[i]);

        if (!was_set)
        {
            ++counter;
        }
    }
}

int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example shows how every second bit of stdgpu::bitset can be set concurrently in a GPU kernel.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n / 2);
    stdgpu::bitset<> bits = stdgpu::bitset<>::createDeviceObject(n);
    stdgpu::atomic<int> counter = stdgpu::atomic<int>::createDeviceObject();

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);

    // d_input : 1, 2, 3, ..., 100

    thrust::copy_if(stdgpu::device_cbegin(d_input),
                    stdgpu::device_cend(d_input),
                    stdgpu::device_begin(d_result),
                    is_odd());

    // d_result : 1, 3, 5, ..., 99

    // bits : 000000..00

    counter.store(0);

    set_bits(d_result, n / 2, bits, counter);

    // bits : 010101...01

    std::cout << "First run: The number of set bits is " << bits.count() << " (" << n / 2 << " expected; "
              << counter.load() << " of those previously unset)" << std::endl;

    counter.store(0);

    set_bits(d_result, n / 2, bits, counter);

    // bits : 010101...01

    std::cout << "Second run: The number of set bits is " << bits.count() << " (" << n / 2 << " expected; "
              << counter.load() << " of those previously unset)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::bitset<>::destroyDeviceObject(bits);
    stdgpu::atomic<int>::destroyDeviceObject(counter);
}
