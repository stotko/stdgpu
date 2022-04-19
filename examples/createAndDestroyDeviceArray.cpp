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

#include <stdgpu/memory.h> // createDeviceArray, copyCreateDevice2HostArray, destroyHostArray, destroyDeviceArray

int
main()
{
    const stdgpu::index_t n = 1000;
    const int default_value = 42;

    int* d_array = createDeviceArray<int>(n, default_value);

    int* h_array = copyCreateDevice2HostArray<int>(d_array, n);

    stdgpu::index_t count = 0;
    for (stdgpu::index_t i = 0; i < n; ++i)
    {
        if (h_array[i] != default_value)
        {
            std::cout << "h_array[" << i << "] != " << default_value << std::endl;
        }
        else
        {
            ++count;
        }
    }

    std::cout << count << "/" << n << " array elements set with default value " << default_value << std::endl;

    destroyHostArray<int>(h_array);
    destroyDeviceArray<int>(d_array);
}
