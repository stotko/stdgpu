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
#include <stdgpu/unordered_map.cuh> // stdgpu::unordered_map

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

struct int_pair_plus
{
    STDGPU_HOST_DEVICE stdgpu::pair<int, int>
    operator()(const stdgpu::pair<int, int>& lhs, const stdgpu::pair<int, int>& rhs) const
    {
        return { lhs.first + rhs.first, lhs.second + rhs.second };
    }
};

void
insert_neighbors(const int* d_result, const stdgpu::index_t n, stdgpu::unordered_map<int, int>& map)
{
#pragma omp parallel for
    for (stdgpu::index_t i = 0; i < n; ++i)
    {
        int num = d_result[i];
        int num_neighborhood[3] = { num - 1, num, num + 1 };

        for (int num_neighbor : num_neighborhood)
        {
            map.emplace(num_neighbor, square()(num_neighbor));
        }
    }
}

int
main()
{
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_map is used to compute a duplicate-free set of numbers.
    //

    const stdgpu::index_t n = 100;

    int* d_input = createDeviceArray<int>(n);
    int* d_result = createDeviceArray<int>(n / 2);
    stdgpu::unordered_map<int, int> map = stdgpu::unordered_map<int, int>::createDeviceObject(n);

    thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);

    // d_input : 1, 2, 3, ..., 100

    thrust::copy_if(stdgpu::device_cbegin(d_input),
                    stdgpu::device_cend(d_input),
                    stdgpu::device_begin(d_result),
                    is_odd());

    // d_result : 1, 3, 5, ..., 99

    insert_neighbors(d_result, n / 2, map);

    // map : 0, 1, 2, 3, ..., 100

    auto range_map = map.device_range();
    stdgpu::pair<int, int> sum =
            thrust::reduce(range_map.begin(), range_map.end(), stdgpu::pair<int, int>(0, 0), int_pair_plus());

    const stdgpu::pair<int, int> sum_closed_form = { n * (n + 1) / 2, n * (n + 1) * (2 * n + 1) / 6 };

    std::cout << "The duplicate-free map of numbers contains " << map.size() << " elements (" << n + 1
              << " expected) and the computed sums are (" << sum.first << ", " << sum.second << ") (("
              << sum_closed_form.first << ", " << sum_closed_form.second << ") expected)" << std::endl;

    destroyDeviceArray<int>(d_input);
    destroyDeviceArray<int>(d_result);
    stdgpu::unordered_map<int, int>::destroyDeviceObject(map);
}
