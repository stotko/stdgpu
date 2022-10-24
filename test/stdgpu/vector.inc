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

#include <stdgpu/algorithm.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/numeric.h>
#include <stdgpu/utility.h>
#include <stdgpu/vector.cuh>
#include <test_memory_utils.h>

class stdgpu_vector : public ::testing::Test
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

template class vector<int>;

// Instantiation of variadic templates emit warnings in CUDA backend
/*
template
STDGPU_DEVICE_ONLY bool
vector<int>::emplace_back<int>(int&&);
*/

template void vector<int>::insert(device_ptr<const int>, device_ptr<int>, device_ptr<int>);

} // namespace stdgpu

TEST_F(stdgpu_vector, empty_container)
{
    stdgpu::vector<int> empty_container;

    EXPECT_TRUE(empty_container.empty());
    EXPECT_TRUE(empty_container.full());
    EXPECT_EQ(empty_container.size(), 0);
    EXPECT_TRUE(empty_container.valid());
}

template <typename T>
class pop_back_vector
{
public:
    explicit pop_back_vector(const stdgpu::vector<T>& pool)
      : _pool(pool)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        _pool.pop_back();
    }

private:
    stdgpu::vector<T> _pool;
};

template <typename Pair>
class pop_back_vector_const_type
{
public:
    explicit pop_back_vector_const_type(const stdgpu::vector<Pair>& pool)
      : _pool(pool)
    {
    }

    inline STDGPU_HOST_DEVICE void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        _pool.pop_back();
    }

private:
    stdgpu::vector<Pair> _pool;
};

template <typename T, typename Values = T>
class push_back_vector
{
public:
    push_back_vector(const stdgpu::vector<T>& pool, Values* values)
      : _pool(pool)
      , _values(values)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _pool.push_back(_values[i]);
    }

private:
    stdgpu::vector<T> _pool;
    Values* _values;
};

template <typename Pair>
class push_back_vector_const_type
{
public:
    push_back_vector_const_type(const stdgpu::vector<Pair>& pool,
                                typename Pair::first_type* firsts,
                                const typename Pair::second_type& second)
      : _pool(pool)
      , _firsts(firsts)
      , _second(second)
    {
    }

    inline STDGPU_HOST_DEVICE void
    operator()(const stdgpu::index_t i)
    {
        _pool.push_back(stdgpu::pair<typename Pair::first_type, typename Pair::second_type>(_firsts[i], _second));
    }

private:
    stdgpu::vector<Pair> _pool;
    typename Pair::first_type* _firsts;
    typename Pair::second_type _second;
};

template <typename T, typename Values = T>
class emplace_back_vector
{
public:
    emplace_back_vector(const stdgpu::vector<T>& pool, Values* values)
      : _pool(pool)
      , _values(values)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _pool.emplace_back(_values[i]);
    }

private:
    stdgpu::vector<T> _pool;
    Values* _values;
};

template <typename Pair>
class emplace_back_vector_const_type
{
public:
    emplace_back_vector_const_type(const stdgpu::vector<Pair>& pool,
                                   typename Pair::first_type* firsts,
                                   const typename Pair::second_type& second)
      : _pool(pool)
      , _firsts(firsts)
      , _second(second)
    {
    }

    inline STDGPU_HOST_DEVICE void
    operator()(const stdgpu::index_t i)
    {
        _pool.emplace_back(_firsts[i], _second);
    }

private:
    stdgpu::vector<Pair> _pool;
    typename Pair::first_type* _firsts;
    typename Pair::second_type _second;
};

void
fill_vector(stdgpu::vector<int>& pool, const stdgpu::index_t N)
{
    ASSERT_GE(N, 0);
    ASSERT_LE(N, pool.capacity());

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector<int>(pool, values));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());
    copyHost2DeviceArray<int>(host_numbers, N, pool.data());

    ASSERT_EQ(pool.size(), N);
    ASSERT_TRUE(pool.valid());
    ASSERT_TRUE((N == 0) ? pool.empty() : !pool.empty());
    ASSERT_TRUE((N == pool.capacity()) ? pool.full() : !pool.full());

    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

void
fill_vector(stdgpu::vector<int>& pool)
{
    fill_vector(pool, pool.capacity());
}

class nondefault_int_vector
{
public:
    nondefault_int_vector() = delete;

    STDGPU_HOST_DEVICE
    nondefault_int_vector(const int x) // NOLINT(hicpp-explicit-conversions)
      : _x(x)
    {
    }

    STDGPU_HOST_DEVICE
    ~nondefault_int_vector() // NOLINT(hicpp-use-equals-default,modernize-use-equals-default)
    {
        // nontrivial destructor
    }

    nondefault_int_vector(const nondefault_int_vector&) = default;
    nondefault_int_vector&
    operator=(const nondefault_int_vector&) = default;

    nondefault_int_vector(nondefault_int_vector&&) = default;
    nondefault_int_vector&
    operator=(nondefault_int_vector&&) = default;

    STDGPU_HOST_DEVICE
    operator int() const // NOLINT(hicpp-explicit-conversions)
    {
        return _x;
    }

private:
    int _x;
};

TEST_F(stdgpu_vector, create_destroy_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, pop_back_some)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = 1000;
    const stdgpu::index_t N_remaining = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N_remaining; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}

TEST_F(stdgpu_vector, pop_back_all)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, pop_back_too_many)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, pop_back_const_type)
{
    using T = stdgpu::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    const float part_second = 2.0F;
    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector_const_type<T>(pool, values, part_second));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::for_each_index(stdgpu::execution::device, N, pop_back_vector_const_type<T>(pool));

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.empty());
    EXPECT_FALSE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, pop_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector<nondefault_int_vector, int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::for_each_index(stdgpu::execution::device, N, pop_back_vector<nondefault_int_vector>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, push_back_some)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = 1000;
    const stdgpu::index_t N_push = N_pop;
    const stdgpu::index_t N_remaining = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1 + N_remaining);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, push_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, push_back_all)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N;
    const stdgpu::index_t N_push = N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, push_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, push_back_too_many)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N;
    const stdgpu::index_t N_push = N_pop + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, push_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        // Only test if all numbers are inside range since N_push - N_pop threads had no chance to insert their numbers
        EXPECT_GE(host_numbers[i], 1);
        EXPECT_LE(host_numbers[i], N_push);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, push_back_const_type)
{
    using T = stdgpu::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    const float part_second = 2.0F;
    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector_const_type<T>(pool, values, part_second));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, push_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector<nondefault_int_vector, int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, emplace_back_some)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = 1000;
    const stdgpu::index_t N_push = N_pop;
    const stdgpu::index_t N_remaining = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1 + N_remaining);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, push_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, emplace_back_all)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N;
    const stdgpu::index_t N_push = N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, emplace_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, emplace_back_too_many)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = N;
    const stdgpu::index_t N_push = N_pop + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    int* values = createDeviceArray<int>(N_push);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N_push, emplace_back_vector<int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        // Only test if all numbers are inside range since N_push - N_pop threads had no chance to insert their numbers
        EXPECT_GE(host_numbers[i], 1);
        EXPECT_LE(host_numbers[i], N_push);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, emplace_back_const_type)
{
    using T = stdgpu::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    const float part_second = 2.0F;
    stdgpu::for_each_index(stdgpu::execution::device, N, emplace_back_vector_const_type<T>(pool, values, part_second));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, emplace_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N, emplace_back_vector<nondefault_int_vector, int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, insert)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_insert = N / 4;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    int* values = createDeviceArray<int>(N_insert);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), N_init + 1);

    pool.insert(pool.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

    ASSERT_EQ(pool.size(), N_init + N_insert);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), pool.size());
    for (stdgpu::index_t i = 0; i < pool.size(); ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, insert_non_end)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_insert = N / 4;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    int* values = createDeviceArray<int>(N_insert);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), N_init + 1);

    pool.insert(pool.device_end() - 1, stdgpu::device_begin(values), stdgpu::device_end(values));

    ASSERT_EQ(pool.size(), N_init);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, insert_too_many)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_insert = N + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    int* values = createDeviceArray<int>(N_insert);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), N_init + 1);

    pool.insert(pool.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

    ASSERT_EQ(pool.size(), N_init);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

TEST_F(stdgpu_vector, erase)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_erase = N / 4;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    pool.erase(pool.device_end() - N_erase, pool.device_end());

    ASSERT_EQ(pool.size(), N_init - N_erase);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), pool.size());
    for (stdgpu::index_t i = 0; i < pool.size(); ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}

TEST_F(stdgpu_vector, erase_non_end)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_erase = N / 4;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    pool.erase(pool.device_end() - N_erase, pool.device_end() - 1);

    ASSERT_EQ(pool.size(), N_init);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, erase_too_many)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_init = N / 2;
    const stdgpu::index_t N_erase = N + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool, N_init);

    pool.erase(pool.device_end() - N_erase, pool.device_end());

    ASSERT_EQ(pool.size(), N_init);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, clear)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    pool.clear();

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, clear_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device, N, push_back_vector<nondefault_int_vector, int>(pool, values));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    pool.clear();

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
    destroyDeviceArray<int>(values);
}

template <typename T>
class simultaneous_push_back_and_pop_back_vector
{
public:
    simultaneous_push_back_and_pop_back_vector(const stdgpu::vector<T>& pool,
                                               const stdgpu::vector<T>& pool_validation,
                                               T* values)
      : _pool(pool)
      , _pool_validation(pool_validation)
      , _values(values)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _pool.push_back(_values[i]);

        stdgpu::pair<T, bool> popped = _pool.pop_back();

        if (popped.second)
        {
            _pool_validation.push_back(popped.first);
        }
    }

private:
    stdgpu::vector<T> _pool;
    stdgpu::vector<T> _pool_validation;
    T* _values;
};

TEST_F(stdgpu_vector, simultaneous_push_back_and_pop_back)
{
    const stdgpu::index_t N = 100000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);
    stdgpu::vector<int> pool_validation = stdgpu::vector<int>::createDeviceObject(N);

    int* values = createDeviceArray<int>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(values), stdgpu::device_end(values), 1);

    stdgpu::for_each_index(stdgpu::execution::device,
                           N,
                           simultaneous_push_back_and_pop_back_vector<int>(pool, pool_validation, values));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    stdgpu::vector<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(values);
}

class at_non_const_vector
{
public:
    explicit at_non_const_vector(const stdgpu::vector<int>& pool)
      : _pool(pool)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        int x = _pool.at(i);
        _pool.at(i) = x * x;
    }

private:
    stdgpu::vector<int> _pool;
};

TEST_F(stdgpu_vector, at_non_const)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N, at_non_const_vector(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], static_cast<int>(i + 1) * static_cast<int>(i + 1));
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}

class access_operator_non_const_vector
{
public:
    explicit access_operator_non_const_vector(const stdgpu::vector<int>& pool)
      : _pool(pool)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        int x = _pool[i];
        _pool[i] = x * x;
    }

private:
    stdgpu::vector<int> _pool;
};

TEST_F(stdgpu_vector, access_operator_non_const)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N, access_operator_non_const_vector(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], static_cast<int>(i + 1) * static_cast<int>(i + 1));
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}

TEST_F(stdgpu_vector, shrink_to_fit_empty)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    ASSERT_EQ(pool.size(), 0);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.capacity() == 0 || pool.capacity() == N); // Capacity may have changed or not
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, shrink_to_fit_full)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), N);
    EXPECT_EQ(pool.capacity(), N);
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, shrink_to_fit)
{
    const stdgpu::index_t N = 10000;
    const stdgpu::index_t N_pop = 100;
    const stdgpu::index_t N_remaining = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    stdgpu::for_each_index(stdgpu::execution::device, N_pop, pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), N_remaining);
    EXPECT_TRUE(pool.capacity() == N_remaining || pool.capacity() == N); // Capacity may have changed or not
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

namespace
{
template <typename T>
class non_const_front_functor
{
public:
    non_const_front_functor(const stdgpu::vector<T>& pool, T* result)
      : _pool(pool)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.front();
    }

private:
    stdgpu::vector<T> _pool;
    T* _result;
};

template <typename T>
class const_front_functor
{
public:
    const_front_functor(const stdgpu::vector<T>& pool, T* result)
      : _pool(pool)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.front();
    }

private:
    const stdgpu::vector<T> _pool;
    T* _result;
};

template <typename T>
T
non_const_front(const stdgpu::vector<T>& pool)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, non_const_front_functor<T>(pool, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
const_front(const stdgpu::vector<T>& pool)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, const_front_functor<T>(pool, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
front(const stdgpu::vector<T>& pool)
{
    T non_const_front_value = non_const_front(pool);
    T const_front_value = const_front(pool);

    EXPECT_EQ(non_const_front_value, const_front_value);

    return non_const_front_value;
}

template <typename T>
class non_const_back_functor
{
public:
    non_const_back_functor(const stdgpu::vector<T>& pool, T* result)
      : _pool(pool)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.back();
    }

private:
    stdgpu::vector<T> _pool;
    T* _result;
};

template <typename T>
class const_back_functor
{
public:
    const_back_functor(const stdgpu::vector<T>& pool, T* result)
      : _pool(pool)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.back();
    }

private:
    const stdgpu::vector<T> _pool;
    T* _result;
};

template <typename T>
T
non_const_back(const stdgpu::vector<T>& pool)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, non_const_back_functor<T>(pool, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
const_back(const stdgpu::vector<T>& pool)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, const_back_functor<T>(pool, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
back(const stdgpu::vector<T>& pool)
{
    T non_const_back_value = non_const_back(pool);
    T const_back_value = const_back(pool);

    EXPECT_EQ(non_const_back_value, const_back_value);

    return non_const_back_value;
}

template <typename T>
class non_const_operator_access_functor
{
public:
    non_const_operator_access_functor(const stdgpu::vector<T>& pool, const stdgpu::index_t index, T* result)
      : _pool(pool)
      , _index(index)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool[_index];
    }

private:
    stdgpu::vector<T> _pool;
    stdgpu::index_t _index;
    T* _result;
};

template <typename T>
class const_operator_access_functor
{
public:
    const_operator_access_functor(const stdgpu::vector<T>& pool, const stdgpu::index_t index, T* result)
      : _pool(pool)
      , _index(index)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool[_index];
    }

private:
    const stdgpu::vector<T> _pool;
    stdgpu::index_t _index;
    T* _result;
};

template <typename T>
T
non_const_operator_access(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, non_const_operator_access_functor<T>(pool, i, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
const_operator_access(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, const_operator_access_functor<T>(pool, i, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
operator_access(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T non_const_operator_access_value = non_const_operator_access(pool, i);
    T const_operator_access_value = const_operator_access(pool, i);

    EXPECT_EQ(non_const_operator_access_value, const_operator_access_value);

    return non_const_operator_access_value;
}

template <typename T>
class non_const_at_functor
{
public:
    non_const_at_functor(const stdgpu::vector<T>& pool, const stdgpu::index_t index, T* result)
      : _pool(pool)
      , _index(index)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.at(_index);
    }

private:
    stdgpu::vector<T> _pool;
    stdgpu::index_t _index;
    T* _result;
};

template <typename T>
class const_at_functor
{
public:
    const_at_functor(const stdgpu::vector<T>& pool, const stdgpu::index_t index, T* result)
      : _pool(pool)
      , _index(index)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = _pool.at(_index);
    }

private:
    const stdgpu::vector<T> _pool;
    stdgpu::index_t _index;
    T* _result;
};

template <typename T>
T
non_const_at(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, non_const_at_functor<T>(pool, i, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
const_at(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T* result = createDeviceArray<T>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, const_at_functor<T>(pool, i, result));

    T host_result;
    copyDevice2HostArray<T>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<T>(result);

    return host_result;
}

template <typename T>
T
at(const stdgpu::vector<T>& pool, const stdgpu::index_t i)
{
    T non_const_at_value = non_const_at(pool, i);
    T const_at_value = const_at(pool, i);

    EXPECT_EQ(non_const_at_value, const_at_value);

    return non_const_at_value;
}
} // namespace

TEST_F(stdgpu_vector, front)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    EXPECT_EQ(front(pool), 1);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, back)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    EXPECT_EQ(back(pool), N);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, operator_access)
{
    const stdgpu::index_t N = 100;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(operator_access(pool, i), i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, at)
{
    const stdgpu::index_t N = 100;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(at(pool, i), i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, data)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    int* non_const_data = pool.data();
    const int* const_data = static_cast<const stdgpu::vector<int>&>(pool).data();

    EXPECT_EQ(non_const_data, const_data);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, device_begin)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    stdgpu::device_ptr<int> non_const_begin = pool.device_begin();
    stdgpu::device_ptr<const int> const_begin = static_cast<const stdgpu::vector<int>&>(pool).device_begin();
    stdgpu::device_ptr<const int> cbegin = static_cast<const stdgpu::vector<int>&>(pool).device_cbegin();

    EXPECT_EQ(non_const_begin, const_begin);
    EXPECT_EQ(const_begin, cbegin);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, device_end)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    stdgpu::device_ptr<int> non_const_end = pool.device_end();
    stdgpu::device_ptr<const int> const_end = static_cast<const stdgpu::vector<int>&>(pool).device_end();
    stdgpu::device_ptr<const int> cend = static_cast<const stdgpu::vector<int>&>(pool).device_cend();

    EXPECT_EQ(non_const_end, const_end);
    EXPECT_EQ(const_end, cend);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, non_const_device_range)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    int* numbers = createDeviceArray<int>(N);

    auto range = pool.device_range();
    stdgpu::copy(stdgpu::execution::device, range.begin(), range.end(), stdgpu::device_begin(numbers));

    int* host_numbers = copyCreateDevice2HostArray<int>(numbers, N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], static_cast<int>(i + 1));
    }

    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(numbers);
    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, const_device_range)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    int* numbers = createDeviceArray<int>(N);

    auto range = static_cast<const stdgpu::vector<int>&>(pool).device_range();
    stdgpu::copy(stdgpu::execution::device, range.begin(), range.end(), stdgpu::device_begin(numbers));

    int* host_numbers = copyCreateDevice2HostArray<int>(numbers, N);
    std::sort(stdgpu::host_begin(host_numbers).get(), stdgpu::host_end(host_numbers).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], static_cast<int>(i + 1));
    }

    destroyHostArray<int>(host_numbers);
    destroyDeviceArray<int>(numbers);
    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, get_allocator)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    stdgpu::vector<int>::allocator_type a = pool.get_allocator();

    int* array = a.allocate(N);
    a.deallocate(array, N);

    stdgpu::vector<int>::destroyDeviceObject(pool);
}

TEST_F(stdgpu_vector, custom_allocator)
{
    test_utils::get_allocator_statistics().reset();

    {
        const stdgpu::index_t N = 10000;

        using Allocator = test_utils::test_device_allocator<int>;
        Allocator a_orig;

        stdgpu::vector<int, Allocator> pool = stdgpu::vector<int, Allocator>::createDeviceObject(N, a_orig);

        stdgpu::vector<int, Allocator>::allocator_type a = pool.get_allocator();

        int* array = a.allocate(N);
        a.deallocate(array, N);

        stdgpu::vector<int, Allocator>::destroyDeviceObject(pool);
    }

    // Account for potential but not guaranteed copy-ellision
    EXPECT_EQ(test_utils::get_allocator_statistics().default_constructions, 1);
    EXPECT_GE(test_utils::get_allocator_statistics().copy_constructions, 12);
    EXPECT_LE(test_utils::get_allocator_statistics().copy_constructions, 20);
    EXPECT_GE(test_utils::get_allocator_statistics().destructions, 13);
    EXPECT_LE(test_utils::get_allocator_statistics().destructions, 21);

    test_utils::get_allocator_statistics().reset();
}
