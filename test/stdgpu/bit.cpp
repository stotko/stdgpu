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
#include <random>
#include <thread>
#include <unordered_set>

#include <test_utils.h>
#include <stdgpu/bit.h>



class stdgpu_bit : public ::testing::Test
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


// Explicit template instantiations
namespace stdgpu
{

template
STDGPU_HOST_DEVICE bool
ispow2<unsigned int>(const unsigned int);

template
STDGPU_HOST_DEVICE unsigned int
mod2<unsigned int>(const unsigned int,
                   const unsigned int);

template
STDGPU_HOST_DEVICE unsigned int
log2pow2<unsigned int>(const unsigned int number);

template
STDGPU_HOST_DEVICE unsigned long long int
log2pow2<unsigned long long int>(const unsigned long long int);

template
STDGPU_HOST_DEVICE int
popcount<unsigned int>(const unsigned int number);

template
STDGPU_HOST_DEVICE int
popcount<unsigned long long int>(const unsigned long long int);

} // namespace stdgpu


void
thread_ispow2_random(const stdgpu::index_t iterations,
                     const std::unordered_set<size_t>& pow2_list)
{
    // Generate true random numbers
    size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<size_t> dist(std::numeric_limits<size_t>::lowest(), std::numeric_limits<size_t>::max());

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        size_t number = dist(rng);

        if (pow2_list.find(number) == pow2_list.end())
        {
            EXPECT_FALSE(stdgpu::ispow2(number));
        }
    }
}


TEST_F(stdgpu_bit, ispow2)
{
    std::unordered_set<size_t> pow2_list;
    for (size_t i = 0; i < 63; ++i)
    {
        size_t pow2_i = static_cast<size_t>(1) << i;

        ASSERT_TRUE(stdgpu::ispow2(pow2_i));

        pow2_list.insert(pow2_i);
    }


    stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 19));

    test_utils::for_each_concurrent_thread(&thread_ispow2_random,
                                           iterations_per_thread,
                                           pow2_list);
}


void
thread_mod2_random(const stdgpu::index_t iterations,
                   const size_t divider)
{
    // Generate true random numbers
    size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<size_t> dist(std::numeric_limits<size_t>::lowest(), std::numeric_limits<size_t>::max());

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        size_t number = dist(rng);
        EXPECT_EQ(stdgpu::mod2(number, divider), number % divider);
    }
}


TEST_F(stdgpu_bit, mod2_random)
{
    const size_t divider = static_cast<size_t>(pow(2, 21));
    stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 19));

    test_utils::for_each_concurrent_thread(&thread_mod2_random,
                                           iterations_per_thread,
                                           divider);
}


TEST_F(stdgpu_bit, mod2_one_positive)
{
    size_t number       = 42;
    size_t divider      = 1;
    EXPECT_EQ(stdgpu::mod2(number, divider), static_cast<size_t>(0));
}


TEST_F(stdgpu_bit, mod2_one_zero)
{
    size_t number       = 0;
    size_t divider      = 1;
    EXPECT_EQ(stdgpu::mod2(number, divider), static_cast<size_t>(0));
}


TEST_F(stdgpu_bit, log2pow2)
{
    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(stdgpu::log2pow2(static_cast<size_t>(1) << i), static_cast<size_t>(i));
    }
}


TEST_F(stdgpu_bit, popcount_zero)
{
    EXPECT_EQ(stdgpu::popcount(static_cast<size_t>(0)), 0);
}


TEST_F(stdgpu_bit, popcount_pow2)
{
    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(stdgpu::popcount(static_cast<size_t>(1) << i), 1);
    }
}


TEST_F(stdgpu_bit, popcount_pow2m1)
{
    for (size_t i = 0; i < std::numeric_limits<size_t>::digits; ++i)
    {
        EXPECT_EQ(stdgpu::popcount((static_cast<size_t>(1) << i) - 1), i);
    }
}

