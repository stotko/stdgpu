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

#include <test_utils.h>
#include <stdgpu/cstdlib.h>



class stdgpu_cstdlib : public ::testing::Test
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


TEST_F(stdgpu_cstdlib, sizedivPow2_zero)
{
    stdgpu::sizediv_t result = stdgpu::sizedivPow2(0, 2);

    EXPECT_EQ(result.quot, 0);
    EXPECT_EQ(result.rem,  0);
}


void
thread_values(const stdgpu::index_t iterations)
{
    size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<size_t> dist_x(1, std::numeric_limits<long long int>::max());
    std::uniform_int_distribution<int> dist_y(1, std::numeric_limits<size_t>::digits);

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        size_t x = dist_x(rng);
        size_t y = (static_cast<size_t>(1) << dist_y(rng));

        EXPECT_EQ(std::lldiv(x, y).quot, stdgpu::sizedivPow2(x, y).quot);
        EXPECT_EQ(std::lldiv(x, y).rem , stdgpu::sizedivPow2(x, y).rem);
    }
}


TEST_F(stdgpu_cstdlib, sizedivPow2_random)
{
    stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 19));

    test_utils::for_each_concurrent_thread(&thread_values,
                                           iterations_per_thread);
}


