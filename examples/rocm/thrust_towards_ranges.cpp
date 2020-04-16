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

#include <stdgpu/atomic.cuh>        // stdgpu::atomic
#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/unordered_set.cuh> // stdgpu::unordered_set



struct square_int
{
    STDGPU_HOST_DEVICE int
    operator()(const int x) const
    {
        return x * x;
    }
};


class atomic_sum
{
    public:
        explicit atomic_sum(stdgpu::atomic<int> sum)
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
    stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n);
    stdgpu::unordered_set<int> set = stdgpu::unordered_set<int>::createDeviceObject(n);
    stdgpu::atomic<int> sum = stdgpu::atomic<int>::createDeviceObject();

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

    sum.store(0);

    auto range_set = set.device_range();
    thrust::for_each(range_set.begin(), range_set.end(),
                     atomic_sum(sum));

    // If thrust had a range interface (maybe in a future release), the above call could also be written in a shorter form:
    //
    // thrust::for_each(set.device_range(),
    //                  atomic_sum(sum));
    //
    // Or the call to device_range may also become an implicit operation in the future.

    std::cout << "The computed sum from i = 1 to " << n << " of i^2 is " << sum.load() << " (" << n * (n + 1) * (2 * n + 1) / 6 << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::unordered_set<int>::destroyDeviceObject(set);
    stdgpu::atomic<int>::destroyDeviceObject(sum);
}


