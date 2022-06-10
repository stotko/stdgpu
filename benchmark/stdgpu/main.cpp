/*
 *  Copyright 2022 Patrick Stotko
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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <stdgpu/device.h>
#include <stdgpu/memory.h>

int
main(int argc, char* argv[])
{
    // Print header
    std::string project_name = "stdgpu";
    std::string project_version = STDGPU_VERSION_STRING;

    const int title_total_width = 57;
    int title_size = static_cast<int>(project_name.size()) + static_cast<int>(project_version.size()) + 1;
    int title_space_left = std::max<int>(1, (title_total_width - title_size) / 2);
    int title_space_right = std::max<int>(1, title_total_width - title_size - title_space_left);

    std::string title = project_name + " " + project_version;
    printf("+---------------------------------------------------------+\n");
    printf("|                                                         |\n");
    printf("|%*s%*s%*s|\n", title_space_left, " ", title_size, title.c_str(), title_space_right, " ");
    printf("|                                                         |\n");
    printf("+---------------------------------------------------------+\n");
    printf("\n");

    stdgpu::print_device_information();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    // Print footer
    printf("\n");
    printf("+---------------------------------------------------------+\n");
    printf("| Memory Usage : #Created / #Destroyed (#Leaks)           |\n");
    printf("|   Device     %6" STDGPU_PRIINDEX64 " / %6" STDGPU_PRIINDEX64 " (%6" STDGPU_PRIINDEX64
           ")                   |\n",
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::device),
           stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::device),
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::device) -
                   stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::device));
    printf("|   Host       %6" STDGPU_PRIINDEX64 " / %6" STDGPU_PRIINDEX64 " (%6" STDGPU_PRIINDEX64
           ")                   |\n",
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::host),
           stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::host),
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::host) -
                   stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::host));
    printf("|   Managed    %6" STDGPU_PRIINDEX64 " / %6" STDGPU_PRIINDEX64 " (%6" STDGPU_PRIINDEX64
           ")                   |\n",
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::managed),
           stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::managed),
           stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::managed) -
                   stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::managed));
    printf("+---------------------------------------------------------+\n");

    return EXIT_SUCCESS;
}
