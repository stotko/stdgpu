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
#include <cmath>
#include <limits>
#include <random>

#include <stdgpu/cmath.h>
#include <test_utils.h>

class stdgpu_cmath : public ::testing::Test
{
protected:
    // Called before each test
    void
    SetUp() override
    {
    }

    // Called after each test
    void
    TearDown() override
    {
    }
};

TEST_F(stdgpu_cmath, abs_zero)
{
    EXPECT_FLOAT_EQ(stdgpu::abs(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(stdgpu::abs(-0.0F), 0.0F);
}

TEST_F(stdgpu_cmath, abs_infinity)
{
    EXPECT_FLOAT_EQ(stdgpu::abs(std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(stdgpu::abs(-std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity());
}

void
thread_positive_values(const stdgpu::index_t iterations)
{
    std::size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        float value = dist(rng);

        EXPECT_EQ(std::abs(value), stdgpu::abs(value));
    }
}

TEST_F(stdgpu_cmath, abs_positive_values)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 21));

    test_utils::for_each_concurrent_thread(&thread_positive_values, iterations_per_thread);
}

void
thread_negative_values(const stdgpu::index_t iterations)
{
    std::size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::lowest(),
                                               -std::numeric_limits<float>::min());

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        float value = dist(rng);

        EXPECT_EQ(std::abs(value), stdgpu::abs(value));
    }
}

TEST_F(stdgpu_cmath, abs_negative_values)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 21));

    test_utils::for_each_concurrent_thread(&thread_negative_values, iterations_per_thread);
}
