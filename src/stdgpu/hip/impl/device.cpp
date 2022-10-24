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

#include <stdgpu/hip/device.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <string>

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
} // namespace detail

namespace stdgpu::hip
{

void
print_device_information()
{
    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess)
    {
        printf("+---------------------------------------------------------+\n");
        printf("|                   Invalid HIP Device                    |\n");
        printf("+---------------------------------------------------------+\n");
        printf("| WARNING: Unable to fetch properties of invalid device!  |\n");
        printf("+---------------------------------------------------------+\n\n");

        return;
    }

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    hipMemGetInfo(&free_memory, &total_memory);

    std::string gpu_name = properties.name;
    const int gpu_name_total_width = 57;
    int gpu_name_size = static_cast<int>(gpu_name.size());
    int gpu_name_space_left = std::max<int>(1, (gpu_name_total_width - gpu_name_size) / 2);
    int gpu_name_space_right = std::max<int>(1, gpu_name_total_width - gpu_name_size - gpu_name_space_left);

    printf("+---------------------------------------------------------+\n");
    printf("|%*s%*s%*s|\n", gpu_name_space_left, " ", gpu_name_size, gpu_name.c_str(), gpu_name_space_right, " ");
    printf("+---------------------------------------------------------+\n");
    printf("| GCN Architecture          :   %4d                      |\n", properties.gcnArch);
    printf("| Clock rate                :   %-6.0f MHz                |\n",
           static_cast<double>(detail::kilo_to_mega_hertz(static_cast<float>(properties.clockRate))));
    printf("| Global Memory             :   %-6.3f GiB / %-6.3f GiB   |\n",
           static_cast<double>(detail::byte_to_gibi_byte(static_cast<float>(free_memory))),
           static_cast<double>(detail::byte_to_gibi_byte(static_cast<float>(total_memory))));
    printf("| Memory Bus Width          :   %-6d Bit                |\n", properties.memoryBusWidth);
    printf("| Multiprocessor (SM) count :   %-6d                    |\n", properties.multiProcessorCount);
    printf("| Warp size                 :   %-6d Threads            |\n", properties.warpSize);
    printf("| L2 Cache                  :   %-6.0f KiB                |\n",
           static_cast<double>(detail::byte_to_kibi_byte(static_cast<float>(properties.l2CacheSize))));
    printf("| Total Constant Memory     :   %-6.0f KiB                |\n",
           static_cast<double>(detail::byte_to_kibi_byte(static_cast<float>(properties.totalConstMem))));
    printf("| Shared Memory per SM      :   %-6.0f KiB                |\n",
           static_cast<double>(
                   detail::byte_to_kibi_byte(static_cast<float>(properties.maxSharedMemoryPerMultiProcessor))));
    printf("| Total Shared Memory       :   %-6.0f KiB                |\n",
           static_cast<double>(detail::byte_to_kibi_byte(
                   static_cast<float>(properties.maxSharedMemoryPerMultiProcessor *
                                      static_cast<std::size_t>(properties.multiProcessorCount)))));
    printf("+---------------------------------------------------------+\n\n");
}

} // namespace stdgpu::hip
