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

#include <stdgpu/unordered_map.cuh>

namespace stdgpu
{

template <>
struct hash<longlong3>
{
    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const longlong3 &key) const
    {
        return key.x * 73856093 ^ key.y * 19349669 ^ key.z * 83492791;
    }
};

} // namespace stdgpu

STDGPU_HOST_DEVICE bool
operator==(const longlong3 &lhs, const longlong3 &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
};

using voxel_map = stdgpu::unordered_map<longlong3, float>;

__global__ void
add_keys_and_values(const stdgpu::index_t n,
                    const longlong3* voxel_idx,
                    const float* voxel_tsd,
                    voxel_map map)
{
    stdgpu::index_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    longlong3 idx = voxel_idx[i];
    float tsd = voxel_tsd[i];

    map.insert(thrust::make_pair(idx, tsd));
}

int main()
{
    stdgpu::index_t n = 3;

    longlong3 voxel_idx_host[n] = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0}};
    longlong3* voxel_idx;
    cudaMalloc(&voxel_idx, sizeof(longlong3)*n);
    cudaMemcpy(voxel_idx, voxel_idx_host, sizeof(longlong3)*n, cudaMemcpyHostToDevice); 

    float voxel_tsd_host[n] = {0.1, 0.002, -0.3};
    float* voxel_tsd;
    cudaMalloc(&voxel_tsd, sizeof(float)*n);
    cudaMemcpy(voxel_tsd, voxel_tsd_host, sizeof(float)*n, cudaMemcpyHostToDevice); 

    voxel_map map = voxel_map::createDeviceObject(1024, n);

    stdgpu::index_t threads = 128;
    stdgpu::index_t blocks = (n + threads - 1) / threads;
    add_keys_and_values<<<blocks, threads>>>(n, voxel_idx, voxel_tsd, map);
    cudaDeviceSynchronize();

    std::cout << "Number of elements is " << map.size() << " (" << 3 << " expected)" << std::endl;

    cudaFree(voxel_idx);
    cudaFree(voxel_tsd);

    
    n = 1;

    longlong3 additional_voxel_idx_host[n] = {{0, 0, 0}};
    cudaMalloc(&voxel_idx, sizeof(longlong3)*n);
    cudaMemcpy(voxel_idx, additional_voxel_idx_host, sizeof(longlong3)*n, cudaMemcpyHostToDevice); 

    float additional_voxel_tsd_host[n] = {0.0};
    cudaMalloc(&voxel_tsd, sizeof(float)*n);
    cudaMemcpy(voxel_tsd, additional_voxel_tsd_host, sizeof(float)*n, cudaMemcpyHostToDevice); 

    threads = 128;
    blocks = (n + threads - 1) / threads;
    add_keys_and_values<<<blocks, threads>>>(n, voxel_idx, voxel_tsd, map);

    std::cout << "Number of elements after overwriting an existing element with a different value is " << map.size() << " (" << 3 << " expected)" << std::endl;

    cudaFree(voxel_idx);
    cudaFree(voxel_tsd);


    voxel_map::destroyDeviceObject(map);
}
