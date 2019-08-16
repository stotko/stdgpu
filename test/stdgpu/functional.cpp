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
#include <unordered_set>

#include <test_utils.h>
#include <stdgpu/functional.h>



class stdgpu_functional : public ::testing::Test
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
void
check_integer()
{
    stdgpu::index_t N = static_cast<stdgpu::index_t>(std::numeric_limits<T>::max()) - static_cast<stdgpu::index_t>(std::numeric_limits<T>::lowest());

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(N);

    stdgpu::hash<T> hasher;
    for (T i = std::numeric_limits<T>::lowest(); i < std::numeric_limits<T>::max(); ++i)
    {
        hashes.insert(hasher(i));
    }

    EXPECT_GT(hashes.size(), 90 / 100 * N);
}


TEST_F(stdgpu_functional, char)
{
    check_integer<char>();
}


TEST_F(stdgpu_functional, signed_char)
{
    check_integer<signed char>();
}


TEST_F(stdgpu_functional, unsigned_char)
{
    check_integer<unsigned char>();
}


TEST_F(stdgpu_functional, short)
{
    check_integer<short>();
}


TEST_F(stdgpu_functional, unsigned_short)
{
    check_integer<unsigned short>();
}


template <typename T>
void
check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    std::default_random_engine rng(test_utils::random_seed());
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(N);

    stdgpu::hash<T> hasher;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        hashes.insert(hasher(dist(rng)));
    }

    EXPECT_GT(hashes.size(), N * 90 / 100);
}


TEST_F(stdgpu_functional, int)
{
    check_integer_random<int>();
}


TEST_F(stdgpu_functional, unsigned_int)
{
    check_integer_random<unsigned int>();
}


TEST_F(stdgpu_functional, long)
{
    check_integer_random<long>();
}


TEST_F(stdgpu_functional, unsigned_long)
{
    check_integer_random<unsigned long>();
}


TEST_F(stdgpu_functional, long_long)
{
    check_integer_random<long long>();
}


TEST_F(stdgpu_functional, unsigned_long_long)
{
    check_integer_random<unsigned long long>();
}


template <typename T>
void
check_floating_point_random()
{
    const stdgpu::index_t N = 1000000;

    std::default_random_engine rng(test_utils::random_seed());
    std::uniform_real_distribution<T> dist(-1e38, 1e38);

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(N);

    stdgpu::hash<T> hasher;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        hashes.insert(hasher(dist(rng)));
    }

    EXPECT_GT(hashes.size(), N * 90 / 100);
}


TEST_F(stdgpu_functional, float)
{
    check_floating_point_random<float>();
}


TEST_F(stdgpu_functional, double)
{
    check_floating_point_random<double>();
}


TEST_F(stdgpu_functional, long_double)
{
    check_floating_point_random<long double>();
}


enum old_enum
{
    zero = 0,
    one = 1,
    two = 2,
    three = 3
};


TEST_F(stdgpu_functional, enum)
{
    std::unordered_set<std::size_t> hashes;
    hashes.reserve(4);

    stdgpu::hash<old_enum> hasher;
    hashes.insert(hasher(zero));
    hashes.insert(hasher(one));
    hashes.insert(hasher(two));
    hashes.insert(hasher(three));

    EXPECT_GT(hashes.size(), 90 / 100 * 4);
}


enum class scoped_enum
{
    zero = 0,
    one = 1,
    two = 2,
    three = 3
};


TEST_F(stdgpu_functional, enum_class)
{
    std::unordered_set<std::size_t> hashes;
    hashes.reserve(4);

    stdgpu::hash<scoped_enum> hasher;
    hashes.insert(hasher(scoped_enum::zero));
    hashes.insert(hasher(scoped_enum::one));
    hashes.insert(hasher(scoped_enum::two));
    hashes.insert(hasher(scoped_enum::three));

    EXPECT_GT(hashes.size(), 90 / 100 * 4);
}


