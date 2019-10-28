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


struct log2pow2_functor
{
    STDGPU_DEVICE_ONLY size_t
    operator()(const size_t i) const
    {
        return stdgpu::cuda::log2pow2(static_cast<unsigned long long>(1) << i);
    }
};

TEST_F(stdgpu_cuda_bit, log2pow2)
{
    size_t* powers = createDeviceArray<size_t>(std::numeric_limits<size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      log2pow2_functor());

    size_t* host_powers = copyCreateDevice2HostArray<size_t>(powers, std::numeric_limits<size_t>::digits);

    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], static_cast<size_t>(i));
    }

    destroyDeviceArray<size_t>(powers);
    destroyHostArray<size_t>(host_powers);
}


struct popcount_pow2
{
    STDGPU_DEVICE_ONLY size_t
    operator()(const size_t i) const
    {
        return stdgpu::cuda::popcount(static_cast<unsigned long long>(1) << i);
    }
};

TEST_F(stdgpu_cuda_bit, popcount_pow2)
{
    size_t* powers = createDeviceArray<size_t>(std::numeric_limits<size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      popcount_pow2());

    size_t* host_powers = copyCreateDevice2HostArray<size_t>(powers, std::numeric_limits<size_t>::digits);

    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], 1);
    }

    destroyDeviceArray<size_t>(powers);
    destroyHostArray<size_t>(host_powers);
}


struct popcount_pow2m1
{
    STDGPU_DEVICE_ONLY size_t
    operator()(const size_t i) const
    {
        return stdgpu::popcount((static_cast<size_t>(1) << i) - 1);
    }
};

TEST_F(stdgpu_cuda_bit, popcount_pow2m1)
{
    size_t* powers = createDeviceArray<size_t>(std::numeric_limits<size_t>::digits);
    thrust::sequence(stdgpu::device_begin(powers), stdgpu::device_end(powers));

    thrust::transform(stdgpu::device_begin(powers), stdgpu::device_end(powers),
                      stdgpu::device_begin(powers),
                      popcount_pow2m1());

    size_t* host_powers = copyCreateDevice2HostArray<size_t>(powers, std::numeric_limits<size_t>::digits);

    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(host_powers[i], i);
    }

    destroyDeviceArray<size_t>(powers);
    destroyHostArray<size_t>(host_powers);
}

