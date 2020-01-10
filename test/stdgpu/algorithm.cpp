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
#include <type_traits>

#include <test_utils.h>
#include <stdgpu/algorithm.h>



class stdgpu_algorithm : public ::testing::Test
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
STDGPU_HOST_DEVICE const int&
min<int>(const int&,
         const int&);

template
STDGPU_HOST_DEVICE const int&
max<int>(const int&,
         const int&);

template
STDGPU_HOST_DEVICE const int&
clamp<int>(const int&,
           const int&,
           const int&);

} // namespace stdgpu


template <typename T>
void check_min_max()
{
    for (T i = std::numeric_limits<T>::lowest(); i < std::numeric_limits<T>::max(); ++i)
    {
        for (T j = std::numeric_limits<T>::lowest(); j < std::numeric_limits<T>::max(); ++j)
        {
            // i = lowest:max-1 , j = lowest:max-1
            EXPECT_EQ(std::min<T>(i, j), stdgpu::min<T>(i, j));
            EXPECT_EQ(std::max<T>(i, j), stdgpu::max<T>(i, j));
        }

        // i = lowest:max-1 , j = max
        EXPECT_EQ(std::min<T>(i, std::numeric_limits<T>::max()), stdgpu::min<T>(i, std::numeric_limits<T>::max()));
        EXPECT_EQ(std::max<T>(i, std::numeric_limits<T>::max()), stdgpu::max<T>(i, std::numeric_limits<T>::max()));
    }

    // i = max , j = max
    EXPECT_EQ(std::min<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()), stdgpu::min<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));
    EXPECT_EQ(std::max<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()), stdgpu::max<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));
}


TEST_F(stdgpu_algorithm, min_max_uint8_t)
{
    check_min_max<std::uint8_t>();
}


TEST_F(stdgpu_algorithm, min_max_int8_t)
{
    check_min_max<std::int8_t>();
}


template <typename T>
void
thread_check_min_max_integer(const stdgpu::index_t iterations)
{
    // Generate true random numbers
    size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        T a = dist(rng);
        T b = dist(rng);

        EXPECT_EQ(std::min<T>(a, b), stdgpu::min<T>(a, b));
        EXPECT_EQ(std::max<T>(a, b), stdgpu::max<T>(a, b));
    }
}

template <typename T>
void check_min_max_random_integer()
{
    stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 19));

    test_utils::for_each_concurrent_thread(&thread_check_min_max_integer<T>,
                                           iterations_per_thread);
}


TEST_F(stdgpu_algorithm, min_max_uint16_t)
{
    check_min_max_random_integer<std::uint16_t>();
}


TEST_F(stdgpu_algorithm, min_max_int16_t)
{
    check_min_max_random_integer<std::int16_t>();
}


TEST_F(stdgpu_algorithm, min_max_uint32_t)
{
    check_min_max_random_integer<std::uint32_t>();
}


TEST_F(stdgpu_algorithm, min_max_int32_t)
{
    check_min_max_random_integer<std::int32_t>();
}


TEST_F(stdgpu_algorithm, min_max_uint64_t)
{
    check_min_max_random_integer<std::uint64_t>();
}


TEST_F(stdgpu_algorithm, min_max_int64_t)
{
    check_min_max_random_integer<std::int64_t>();
}


template <typename T>
T
random_float(std::uniform_real_distribution<T>& dist,
             std::default_random_engine& rng,
             std::uniform_real_distribution<T>& flip)
{
    T result = dist(rng);

    return flip(rng) < T(0.5) ? result : -result;
}

template <typename T>
void
thread_check_min_max_float(const stdgpu::index_t iterations)
{
    // Generate true random numbers
    size_t seed = test_utils::random_thread_seed();

    std::default_random_engine rng(seed);
    std::uniform_real_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    std::uniform_real_distribution<T> flip(T(0), T(1));

    for (stdgpu::index_t i = 0; i < iterations; ++i)
    {
        T a = random_float(dist, rng, flip);
        T b = random_float(dist, rng, flip);

        EXPECT_EQ(std::min<T>(a, b), stdgpu::min<T>(a, b));
        EXPECT_EQ(std::max<T>(a, b), stdgpu::max<T>(a, b));
    }
}

template <typename T>
void check_min_max_random_float()
{
    stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 19));

    test_utils::for_each_concurrent_thread(&thread_check_min_max_float<T>,
                                           iterations_per_thread);
}


TEST_F(stdgpu_algorithm, min_max_float)
{
    check_min_max_random_float<float>();
}


TEST_F(stdgpu_algorithm, min_max_double)
{
    check_min_max_random_float<double>();
}



