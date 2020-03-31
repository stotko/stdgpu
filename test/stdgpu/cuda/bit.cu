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

#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <stdgpu/bit.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_cuda_bit : public ::testing::Test
{
    protected:
        // Called before each test
        virtual void SetUp()
        {

        }

        // Called after each test
        virtual void TearDown()
        {

        }

};


struct bit_width_functor
{
    STDGPU_DEVICE_ONLY std::size_t
    operator()(const std::size_t i) const
    {
        return stdgpu::cuda::bit_width(static_cast<unsigned long long>(1) << i);
    }
};

TEST_F(stdgpu_cuda_bit, bit_width)
{
    std::size_t* powers = createDeviceArray<std::size_t>(std::numeric_limits<std::size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      bit_width_functor());

    std::size_t* host_powers = copyCreateDevice2HostArray<std::size_t>(powers, std::numeric_limits<std::size_t>::digits);

    for (std::size_t i = 0; i < std::numeric_limits<std::size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], static_cast<std::size_t>(i + 1));
    }

    destroyDeviceArray<std::size_t>(powers);
    destroyHostArray<std::size_t>(host_powers);
}


struct popcount_pow2
{
    STDGPU_DEVICE_ONLY std::size_t
    operator()(const std::size_t i) const
    {
        return stdgpu::cuda::popcount(static_cast<unsigned long long>(1) << i);
    }
};

TEST_F(stdgpu_cuda_bit, popcount_pow2)
{
    std::size_t* powers = createDeviceArray<std::size_t>(std::numeric_limits<std::size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      popcount_pow2());

    std::size_t* host_powers = copyCreateDevice2HostArray<std::size_t>(powers, std::numeric_limits<std::size_t>::digits);

    for (std::size_t i = 0; i < std::numeric_limits<std::size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], 1);
    }

    destroyDeviceArray<std::size_t>(powers);
    destroyHostArray<std::size_t>(host_powers);
}


struct popcount_pow2m1
{
    STDGPU_DEVICE_ONLY std::size_t
    operator()(const std::size_t i) const
    {
        return stdgpu::popcount((static_cast<std::size_t>(1) << i) - 1);
    }
};

TEST_F(stdgpu_cuda_bit, popcount_pow2m1)
{
    std::size_t* powers = createDeviceArray<std::size_t>(std::numeric_limits<std::size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      popcount_pow2m1());

    std::size_t* host_powers = copyCreateDevice2HostArray<std::size_t>(powers, std::numeric_limits<std::size_t>::digits);

    for (std::size_t i = 0; i < std::numeric_limits<std::size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], i);
    }

    destroyDeviceArray<std::size_t>(powers);
    destroyHostArray<std::size_t>(host_powers);
}

