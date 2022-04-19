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
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <stdgpu/atomic.cuh> // stdgpu::atomic
#include <stdgpu/iterator.h> // device_begin, device_end
#include <stdgpu/memory.h>   // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h> // STDGPU_HOST_DEVICE

struct square_int
{
    STDGPU_HOST_DEVICE int
    operator()(const int x) const
    {
        return x * x;
    }
};

class atomic_add
{
public:
    explicit atomic_add(const stdgpu::atomic<int>& sum)
      : _sum(sum)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const int x)
    {
        _sum.fetch_add(x);
    }

private:
    stdgpu::atomic<int> _sum;
};

int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::atomic can be used to compute a sum of numbers by atomic addition.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n);
    stdgpu::atomic<int> sum = stdgpu::atomic<int>::createDeviceObject();

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);

    // d_input : 1, 2, 3, ..., 100

    thrust::transform(stdgpu::device_cbegin(d_input),
                      stdgpu::device_cend(d_input),
                      stdgpu::device_begin(d_result),
                      square_int());

    // d_result : 1, 4, 9, ..., 10000

    sum.store(0);

    thrust::for_each(stdgpu::device_cbegin(d_result), stdgpu::device_cend(d_result), atomic_add(sum));

    const int sum_closed_form = n * (n + 1) * (2 * n + 1) / 6;

    std::cout << "The computed sum from i = 1 to " << n << " of i^2 is " << sum.load() << " (" << sum_closed_form
              << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::atomic<int>::destroyDeviceObject(sum);
}
