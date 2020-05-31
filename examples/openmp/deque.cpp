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

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/deque.cuh>         // stdgpu::deque



struct is_odd
{
    STDGPU_HOST_DEVICE bool
    operator()(const int x) const
    {
        return x % 2 == 1;
    }
};


void
insert_neighbors_with_duplicates(const int* d_input,
                                 const stdgpu::index_t n,
                                 stdgpu::deque<int>& deq)
{
    #pragma omp parallel for
    for (stdgpu::index_t i = 0; i < n; ++i)
    {
        int num = d_input[i];
        int num_neighborhood[3] = { num - 1, num, num + 1 };

        is_odd odd;
        for (int num_neighbor : num_neighborhood)
        {
            if (odd(num_neighbor))
            {
                deq.push_back(num_neighbor);
            }
            else
            {
                deq.push_front(num_neighbor);
            }
        }
    }
}


int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::deque is used to compute a set of duplicated numbers.
    // Every number is contained 3 times, except for the first and last one.
    // Furthermore, even numbers are put into the front, whereas odd number are put into the back.
    //

    stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    stdgpu::deque<int> deq = stdgpu::deque<int>::createDeviceObject(3 * n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input),
                     1);

    // d_input : 1, 2, 3, ..., 100

    insert_neighbors_with_duplicates(d_input, n, deq);

    // deq : 0, 1, 1, 2, 2, 2, 3, 3, 3,  ..., 99, 99, 99, 100, 100, 101

    auto range_deq = deq.device_range();
    int sum = thrust::reduce(range_deq.begin(), range_deq.end(),
                             0,
                             thrust::plus<int>());

    const int sum_closed_form = 3 * (n * (n + 1) / 2);

    std::cout << "The set of duplicated numbers contains " << deq.size() << " elements (" << 3 * n << " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    stdgpu::deque<int>::destroyDeviceObject(deq);
}


