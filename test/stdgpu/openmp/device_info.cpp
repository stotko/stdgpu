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

#include <stdgpu/openmp/device_info.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>



namespace stdgpu
{

namespace openmp
{

void
print_device_information()
{
    const std::string gpu_name = "Built-in CPU";
    const int gpu_name_total_width = 57;
    int gpu_name_size = static_cast<int>(gpu_name.size());
    int gpu_name_space_left  = std::max<int>(1, (gpu_name_total_width - gpu_name_size) / 2);
    int gpu_name_space_right = std::max<int>(1, gpu_name_total_width - gpu_name_size - gpu_name_space_left);

    printf( "+---------------------------------------------------------+\n" );
    printf( "|%*s%*s%*s|\n", gpu_name_space_left, " ", gpu_name_size, gpu_name.c_str(), gpu_name_space_right, " ");
    printf( "+---------------------------------------------------------+\n" );
    printf( "| Logical Processor count   :   %-6d                    |\n", std::thread::hardware_concurrency() );
    printf( "+---------------------------------------------------------+\n\n" );
}

} // namespace openmp

} // namespace stdgpu


