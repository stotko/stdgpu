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

#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE



class Image
{
    public:
        Image() = default;

        static Image
        createDeviceObject(const stdgpu::index_t width,
                           const stdgpu::index_t height)
        {
            Image result;

            result._values = createDeviceArray<std::uint8_t>(width * height);
            result._width = width;
            result._height = height;

            return result;
        }

        static void
        destroyDeviceObject(Image& device_object)
        {
            destroyDeviceArray<std::uint8_t>(device_object._values);
            device_object._width = 0;
            device_object._height = 0;
        }

        // Further (static) member functions ...

        STDGPU_HOST_DEVICE std::uint8_t&
        operator()(const stdgpu::index_t x,
                   const stdgpu::index_t y)
        {
            return _values[y * _width + x];
        }

        STDGPU_HOST_DEVICE stdgpu::index_t
        width() const
        {
            return _width;
        }

        STDGPU_HOST_DEVICE stdgpu::index_t
        height() const
        {
            return _height;
        }

    private:
        std::uint8_t* _values = nullptr;
        stdgpu::index_t _width = 0;
        stdgpu::index_t _height = 0;
};


__global__ void
fill_image(Image d_image)
{
    stdgpu::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    stdgpu::index_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= d_image.width() || j >= d_image.height()) return;

    std::uint8_t value = (i * i + j * j) % (1 << 8);

    d_image(i, j) = value;
}


int
main()
{
    Image d_image = Image::createDeviceObject(1920, 1080);

    dim3 threads(32, 8);
    dim3 blocks((d_image.width() + threads.x - 1) / threads.x, (d_image.height() + threads.y - 1) / threads.y);

    fill_image<<< blocks, threads >>>(d_image);
    cudaDeviceSynchronize();

    Image::destroyDeviceObject(d_image);
}


