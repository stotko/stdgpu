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

#include <limits>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <test_utils.h>
#include <stdgpu/atomic.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_cuda_atomic : public ::testing::Test
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



template <typename T>
struct subtract
{
    T* value;

    subtract(T* value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        atomicSub(value, x);
    }
};


TEST_F(stdgpu_cuda_atomic, unsigned_long_long_int_sub)
{
    const stdgpu::index64_t N = 10000000;

    // Create sequence
    unsigned long long int* numbers = createDeviceArray<unsigned long long int>(N);

    thrust::sequence(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     static_cast<unsigned long long int>(1));

    unsigned long long int* sum = createDeviceArray<unsigned long long int>(1, N * (N + 1) / 2);


    thrust::for_each(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     subtract<unsigned long long int>(sum));


    unsigned long long int* host_sum = copyCreateDevice2HostArray(sum, 1);

    EXPECT_EQ(*host_sum, static_cast<unsigned long long int>(0));


    destroyDeviceArray<unsigned long long int>(numbers);
    destroyDeviceArray<unsigned long long int>(sum);
    destroyHostArray<unsigned long long int>(host_sum);
}


TEST_F(stdgpu_cuda_atomic, unsigned_long_long_int_sub_zero_pattern)
{
    unsigned long long int* number  = createDeviceArray<unsigned long long int>(1, 0);  // zero pattern
    unsigned long long int* sum     = createDeviceArray<unsigned long long int>(1, 42);


    thrust::for_each(stdgpu::device_begin(number), stdgpu::device_end(number),
                     subtract<unsigned long long int>(sum));


    unsigned long long int* host_sum = copyCreateDevice2HostArray(sum, 1);

    EXPECT_EQ(*host_sum, static_cast<unsigned long long int>(42));


    destroyDeviceArray<unsigned long long int>(number);
    destroyDeviceArray<unsigned long long int>(sum);
    destroyHostArray<unsigned long long int>(host_sum);
}


TEST_F(stdgpu_cuda_atomic, unsigned_long_long_int_sub_one_pattern)
{
    unsigned long long int* number  = createDeviceArray<unsigned long long int>(1, std::numeric_limits<unsigned long long int>::max()); // one pattern
    unsigned long long int* sum     = createDeviceArray<unsigned long long int>(1, 42);


    thrust::for_each(stdgpu::device_begin(number), stdgpu::device_end(number),
                     subtract<unsigned long long int>(sum));


    unsigned long long int* host_sum = copyCreateDevice2HostArray(sum, 1);

    EXPECT_EQ(*host_sum, static_cast<unsigned long long int>(43));


    destroyDeviceArray<unsigned long long int>(number);
    destroyDeviceArray<unsigned long long int>(sum);
    destroyHostArray<unsigned long long int>(host_sum);
}


TEST_F(stdgpu_cuda_atomic, float_sub)
{
    const stdgpu::index64_t N = 5000;

    // Create sequence
    float* numbers = createDeviceArray<float>(N);

    thrust::sequence(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     1.0f);

    float* sum = createDeviceArray<float>(1, static_cast<float>(N * (N + 1) / 2));


    thrust::for_each(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     subtract<float>(sum));


    float* host_sum = copyCreateDevice2HostArray(sum, 1);

    EXPECT_FLOAT_EQ(*host_sum, 0.0f);


    destroyDeviceArray<float>(numbers);
    destroyDeviceArray<float>(sum);
    destroyHostArray<float>(host_sum);
}


struct random_float
{
    stdgpu::index_t seed;
    float min, max;

    STDGPU_HOST_DEVICE
    random_float(const stdgpu::index_t seed,
                 const float min,
                 const float max)
        : seed(seed),
          min(min),
          max(max)
    {

    }

    STDGPU_HOST_DEVICE float
    operator()(const stdgpu::index_t n) const
    {
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> dist(min, max);
        rng.discard(n);

        return dist(rng);
    }
};


struct find_min
{
    float* value;

    find_min(float* value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const float x)
    {
        atomicMin(value, x);
    }
};


struct find_max
{
    float* value;

    find_max(float* value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const float x)
    {
        atomicMax(value, x);
    }
};


TEST_F(stdgpu_cuda_atomic, float_min)
{
    const stdgpu::index64_t N = 10000000;
    // thrust::uniform_real_distribution is not stable with std::numeric_limits<float>::{lowest(), max()}
    const float global_min = -1e38f;
    const float global_max =  1e38f;

    // Create random numbers
    float* numbers = createDeviceArray<float>(N);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0),
                      thrust::counting_iterator<stdgpu::index_t>(N),
                      stdgpu::device_begin(numbers),
                      random_float(test_utils::random_seed(),
                                           global_min,
                                           global_max));

    float* min = createDeviceArray<float>(1, std::numeric_limits<float>::max());


   thrust::for_each(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     find_min(min));


    float* host_min     = copyCreateDevice2HostArray(min,     1);
    float* host_numbers = copyCreateDevice2HostArray(numbers, N);


    bool min_found = false;
    for (stdgpu::index64_t i = 0; i < N; ++i)
    {
        // min <= numbers[i]
        EXPECT_LE(*host_min, host_numbers[i]);

        // min in numbers
        if (*host_min == host_numbers[i])
        {
            min_found = true;
        }
    }
    EXPECT_TRUE(min_found);


    destroyDeviceArray<float>(numbers);
    destroyDeviceArray<float>(min);
    destroyHostArray<float>(host_numbers);
    destroyHostArray<float>(host_min);
}


TEST_F(stdgpu_cuda_atomic, float_max)
{
    const stdgpu::index64_t N = 10000000;
    // thrust::uniform_real_distribution is not stable with std::numeric_limits<float>::{lowest(), max()}
    const float global_min = -1e38f;
    const float global_max =  1e38f;

    // Create random numbers
    float* numbers = createDeviceArray<float>(N);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0),
                      thrust::counting_iterator<stdgpu::index_t>(N),
                      stdgpu::device_begin(numbers),
                      random_float(test_utils::random_seed(),
                                           global_min,
                                           global_max));

    float* max = createDeviceArray<float>(1, std::numeric_limits<float>::lowest());


    thrust::for_each(stdgpu::device_begin(numbers), stdgpu::device_end(numbers),
                     find_max(max));


    float* host_max     = copyCreateDevice2HostArray(max,     1);
    float* host_numbers = copyCreateDevice2HostArray(numbers, N);


    bool max_found = false;
    for (stdgpu::index64_t i = 0; i < N; ++i)
    {
        // max >= numbers[i]
        EXPECT_GE(*host_max, host_numbers[i]);

        // max in numbers
        if (*host_max == host_numbers[i])
        {
            max_found = true;
        }
    }
    EXPECT_TRUE(max_found);


    destroyDeviceArray<float>(numbers);
    destroyDeviceArray<float>(max);
    destroyHostArray<float>(host_numbers);
    destroyHostArray<float>(host_max);
}


