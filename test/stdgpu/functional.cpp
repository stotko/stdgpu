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

#include <stdgpu/functional.h>
#include <test_utils.h>

class stdgpu_functional : public ::testing::Test
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

// Explicit template instantiations
namespace stdgpu
{

// Instantiation of specialized templates emit no-effect warnings with Clang
/*
template
struct hash<bool>;

template
struct hash<char>;

template
struct hash<signed char>;

template
struct hash<unsigned char>;

template
struct hash<wchar_t>;

template
struct hash<char16_t>;

template
struct hash<char32_t>;

template
struct hash<short>;

template
struct hash<unsigned short>;

template
struct hash<int>;

template
struct hash<unsigned int>;

template
struct hash<long>;

template
struct hash<unsigned long>;

template
struct hash<long long>;

template
struct hash<unsigned long long>;

template
struct hash<float>;

template
struct hash<double>;

template
struct hash<long double>;
*/

template struct plus<int>;

template struct logical_and<int>;

template struct equal_to<int>;

template struct bit_not<unsigned int>;

} // namespace stdgpu

template <typename T>
void
hash_check_integer()
{
    stdgpu::index_t N = static_cast<stdgpu::index_t>(std::numeric_limits<T>::max()) -
                        static_cast<stdgpu::index_t>(std::numeric_limits<T>::lowest());

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(static_cast<std::size_t>(N));

    stdgpu::hash<T> hasher;
    for (T i = std::numeric_limits<T>::lowest(); i < std::numeric_limits<T>::max(); ++i)
    {
        hashes.insert(hasher(i));
    }

    EXPECT_GT(static_cast<stdgpu::index_t>(hashes.size()), N * 90 / 100);
}

TEST_F(stdgpu_functional, hash_char)
{
    hash_check_integer<char>();
}

TEST_F(stdgpu_functional, hash_signed_char)
{
    hash_check_integer<signed char>();
}

TEST_F(stdgpu_functional, hash_unsigned_char)
{
    hash_check_integer<unsigned char>();
}

TEST_F(stdgpu_functional, short)
{
    hash_check_integer<short>();
}

TEST_F(stdgpu_functional, hash_unsigned_short)
{
    hash_check_integer<unsigned short>();
}

template <typename T>
void
hash_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(static_cast<std::size_t>(N));

    stdgpu::hash<T> hasher;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        hashes.insert(hasher(dist(rng)));
    }

    EXPECT_GT(static_cast<stdgpu::index_t>(hashes.size()), N * 90 / 100);
}

TEST_F(stdgpu_functional, hash_int)
{
    hash_check_integer_random<int>();
}

TEST_F(stdgpu_functional, hash_unsigned_int)
{
    hash_check_integer_random<unsigned int>();
}

TEST_F(stdgpu_functional, hash_long)
{
    hash_check_integer_random<long>();
}

TEST_F(stdgpu_functional, hash_unsigned_long)
{
    hash_check_integer_random<unsigned long>();
}

TEST_F(stdgpu_functional, hash_long_long)
{
    hash_check_integer_random<long long>();
}

TEST_F(stdgpu_functional, hash_unsigned_long_long)
{
    hash_check_integer_random<unsigned long long>();
}

template <typename T>
void
hash_check_floating_point_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    const T bound = static_cast<T>(1e38);
    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_real_distribution<T> dist(-bound, bound);

    std::unordered_set<std::size_t> hashes;
    hashes.reserve(static_cast<std::size_t>(N));

    stdgpu::hash<T> hasher;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        hashes.insert(hasher(dist(rng)));
    }

    EXPECT_GT(static_cast<stdgpu::index_t>(hashes.size()), N * 90 / 100);
}

TEST_F(stdgpu_functional, hash_float)
{
    hash_check_floating_point_random<float>();
}

TEST_F(stdgpu_functional, hash_double)
{
    hash_check_floating_point_random<double>();
}

TEST_F(stdgpu_functional, hash_long_double)
{
    hash_check_floating_point_random<long double>();
}

enum old_enum
{
    zero = 0,
    one = 1,
    two = 2,
    three = 3
};

// cppcheck-suppress syntaxError
TEST_F(stdgpu_functional, hash_enum)
{
    std::unordered_set<std::size_t> hashes;
    hashes.reserve(4);

    stdgpu::hash<old_enum> hasher;
    hashes.insert(hasher(zero));
    hashes.insert(hasher(one));
    hashes.insert(hasher(two));
    hashes.insert(hasher(three));

    EXPECT_GT(static_cast<stdgpu::index_t>(hashes.size()), 4 * 90 / 100);
}

enum class scoped_enum
{
    zero = 0,
    one = 1,
    two = 2,
    three = 3
};

TEST_F(stdgpu_functional, hash_enum_class)
{
    std::unordered_set<std::size_t> hashes;
    hashes.reserve(4);

    stdgpu::hash<scoped_enum> hasher;
    hashes.insert(hasher(scoped_enum::zero));
    hashes.insert(hasher(scoped_enum::one));
    hashes.insert(hasher(scoped_enum::two));
    hashes.insert(hasher(scoped_enum::three));

    EXPECT_GT(static_cast<stdgpu::index_t>(hashes.size()), 4 * 90 / 100);
}

template <typename T>
void
identity_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::identity identity_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value = dist(rng);
        EXPECT_EQ(identity_function(value), value);
    }
}

TEST_F(stdgpu_functional, identity)
{
    identity_check_integer_random<int>();
}

template <typename T>
void
plus_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::plus<T> plus_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value_1 = dist(rng);
        T value_2 = dist(rng);
        EXPECT_EQ(plus_function(value_1, value_2), value_1 + value_2);
    }
}

TEST_F(stdgpu_functional, plus_int)
{
    plus_check_integer_random<int>();
}

template <typename T, typename U>
void
plus_transparent_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::plus<> plus_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value_1 = dist(rng);
        U value_2 = static_cast<U>(dist(rng));
        EXPECT_EQ(plus_function(value_1, value_2), value_1 + value_2);
    }
}

TEST_F(stdgpu_functional, plus_transparent)
{
    plus_transparent_check_integer_random<int, long>();
}

template <typename T>
void
logical_and_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::logical_and<T> logical_and_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value_1 = dist(rng);
        T value_2 = dist(rng);
        EXPECT_EQ(logical_and_function(value_1, value_2), value_1 && value_2);
    }
}

TEST_F(stdgpu_functional, logical_and_int)
{
    logical_and_check_integer_random<int>();
}

template <typename T, typename U>
void
logical_and_transparent_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::logical_and<> logical_and_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value_1 = dist(rng);
        U value_2 = static_cast<U>(dist(rng));
        EXPECT_EQ(logical_and_function(value_1, value_2), value_1 && value_2);
    }
}

TEST_F(stdgpu_functional, logical_and_transparent)
{
    logical_and_transparent_check_integer_random<int, long>();
}

template <typename T>
void
equal_to_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::equal_to<T> equal_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value = dist(rng);
        EXPECT_TRUE(equal_function(value, value));
    }
}

TEST_F(stdgpu_functional, equal_to_int)
{
    equal_to_check_integer_random<int>();
}

template <typename T, typename U>
void
equal_to_transparent_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::equal_to<> equal_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value_1 = dist(rng);
        U value_2 = static_cast<U>(value_1);
        EXPECT_TRUE(equal_function(value_1, value_2));
    }
}

TEST_F(stdgpu_functional, equal_to_transparent)
{
    equal_to_transparent_check_integer_random<int, long>();
}

template <typename T>
void
bit_not_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::bit_not<T> bit_not_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value = dist(rng);
        EXPECT_EQ(bit_not_function(value), ~value);
    }
}

TEST_F(stdgpu_functional, bit_not_unsigned_int)
{
    bit_not_check_integer_random<unsigned int>();
}

template <typename T>
void
bit_not_transparent_check_integer_random()
{
    const stdgpu::index_t N = 1000000;

    // Generate true random numbers
    std::size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

    stdgpu::bit_not<> bit_not_function;
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        T value = dist(rng);
        EXPECT_EQ(bit_not_function(value), ~value);
    }
}

TEST_F(stdgpu_functional, bit_not_transparent)
{
    bit_not_transparent_check_integer_random<unsigned int>();
}
