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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cuda_runtime_api.h>

#include <stdgpu/config.h>
#include <stdgpu/memory.h>



namespace detail
{
    float
    kilo_to_mega_hertz(const float kilo_hertz)
    {
        return kilo_hertz / 1000.0f;
    }

    float
    byte_to_kibi_byte(const float byte)
    {
        return byte / 1024.0f;
    }

    float
    byte_to_gibi_byte(const float byte)
    {
        return byte / (1024.0f * 1024.0f * 1024.0f);
    }
}


GTEST_API_ int
main(int argc, char* argv[])
{
    // Print header
    std::string project_name = "stdgpu";
    std::string project_version = STDGPU_VERSION_STRING;

    int title_total_width = 57;
    int title_size = static_cast<int>(project_name.size()) + static_cast<int>(project_version.size()) + 1;
    int title_space_left  = std::max<int>(1, (title_total_width - title_size) / 2);
    int title_space_right = std::max<int>(1, title_total_width - title_size - title_space_left);

    std::string title = project_name + " " + project_version;
    printf( "+---------------------------------------------------------+\n" );
    printf( "|                                                         |\n");
    printf( "|%*s%*s%*s|\n", title_space_left, " ", title_size, title.c_str(), title_space_right, " ");
    printf( "|                                                         |\n");
    printf( "+---------------------------------------------------------+\n" );
    printf("\n");



    // Print GPU information
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, 0 );

    size_t free_memory  = 0;
    size_t total_memory = 0;
    cudaMemGetInfo(&free_memory, &total_memory);

    std::string gpu_name = properties.name;
    int gpu_name_total_width = 57;
    int gpu_name_size = static_cast<int>(gpu_name.size());
    int gpu_name_space_left  = std::max<int>(1, (gpu_name_total_width - gpu_name_size) / 2);
    int gpu_name_space_right = std::max<int>(1, gpu_name_total_width - gpu_name_size - gpu_name_space_left);

    printf( "+---------------------------------------------------------+\n" );
    printf( "|%*s%*s%*s|\n", gpu_name_space_left, " ", gpu_name_size, gpu_name.c_str(), gpu_name_space_right, " ");
    printf( "+---------------------------------------------------------+\n" );
    printf( "| Compute Capability        :   %1d.%1d                       |\n", properties.major, properties.minor );
    printf( "| Clock rate                :   %-6.0f MHz                |\n", detail::kilo_to_mega_hertz(properties.clockRate));
    printf( "| Global Memory             :   %-6.3f GiB / %-6.3f GiB   |\n", detail::byte_to_gibi_byte(free_memory), detail::byte_to_gibi_byte(total_memory));
    printf( "| Memory Bus Width          :   %-6d Bit                |\n", properties.memoryBusWidth );
    printf( "| Multiprocessor (SM) count :   %-6d                    |\n", properties.multiProcessorCount );
    printf( "| Warp size                 :   %-6d Threads            |\n", properties.warpSize );
    printf( "| L2 Cache                  :   %-6.0f KiB                |\n", detail::byte_to_kibi_byte(properties.l2CacheSize));
    printf( "| Total Constant Memory     :   %-6.0f KiB                |\n", detail::byte_to_kibi_byte(properties.totalConstMem));
    printf( "| Shared Memory per SM      :   %-6.0f KiB                |\n", detail::byte_to_kibi_byte(properties.sharedMemPerMultiprocessor));
    printf( "| Total Shared Memory       :   %-6.0f KiB                |\n", detail::byte_to_kibi_byte(properties.sharedMemPerMultiprocessor * static_cast<size_t>(properties.multiProcessorCount)));
    printf( "+---------------------------------------------------------+\n\n" );



    // Initialize gtest framework
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    int result = RUN_ALL_TESTS();



    // Print footer
    printf("\n");
    printf( "+---------------------------------------------------------+\n" );
    printf( "| Memory Usage : #Created / #Destroyed (#Leaks)           |\n");
    printf( "|   Device     %6ld / %6ld (%6ld)                   |\n", stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::device),
                                                                       stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::device),
                                                                       stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::device) - stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::device));
    printf( "|   Host       %6ld / %6ld (%6ld)                   |\n", stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::host),
                                                                       stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::host),
                                                                       stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::host) - stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::host));
    printf( "|   Managed    %6ld / %6ld (%6ld)                   |\n", stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::managed),
                                                                       stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::managed),
                                                                       stdgpu::get_allocation_count(stdgpu::dynamic_memory_type::managed) - stdgpu::get_deallocation_count(stdgpu::dynamic_memory_type::managed));
    printf( "+---------------------------------------------------------+\n" );



    return result;
}


