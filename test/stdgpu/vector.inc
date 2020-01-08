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

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/vector.cuh>



class stdgpu_vector : public ::testing::Test
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
class vector<int>;

// Instantiation of variadic templates emit warnings in CUDA backend
/*
template
STDGPU_DEVICE_ONLY bool
vector<int>::emplace_back<int>(int&&);
*/

} // namespace stdgpu


template <typename T>
struct pop_back_vector
{
    stdgpu::vector<T> pool;

    pop_back_vector(stdgpu::vector<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(STDGPU_MAYBE_UNUSED const T x)
    {
        pool.pop_back();
    }
};

template <typename Pair>
struct pop_back_vector_const_type
{
    stdgpu::vector<Pair> pool;

    pop_back_vector_const_type(stdgpu::vector<Pair> pool)
        : pool(pool)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(STDGPU_MAYBE_UNUSED const typename Pair::first_type& first)
    {
        pool.pop_back();
    }
};


template <typename T>
struct push_back_vector
{
    stdgpu::vector<T> pool;

    push_back_vector(stdgpu::vector<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.push_back(x);
    }
};

template <typename Pair>
struct push_back_vector_const_type
{
    stdgpu::vector<Pair> pool;
    typename Pair::second_type second;

    push_back_vector_const_type(stdgpu::vector<Pair> pool,
                                const typename Pair::second_type& second)
        : pool(pool),
          second(second)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(const typename Pair::first_type& first)
    {
        pool.push_back(thrust::make_pair(first, second));
    }
};


template <typename T>
struct emplace_back_vector
{
    stdgpu::vector<T> pool;

    emplace_back_vector(stdgpu::vector<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.emplace_back(x);
    }
};

template <typename Pair>
struct emplace_back_vector_const_type
{
    stdgpu::vector<Pair> pool;
    typename Pair::second_type second;

    emplace_back_vector_const_type(stdgpu::vector<Pair> pool,
                                   const typename Pair::second_type& second)
        : pool(pool),
          second(second)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(const typename Pair::first_type& first)
    {
        pool.emplace_back(first, second);
    }
};


void
fill_vector(stdgpu::vector<int> pool)
{
    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(pool.capacity() + init),
                     push_back_vector<int>(pool));

    thrust::sort(stdgpu::device_begin(pool), stdgpu::device_end(pool));

    ASSERT_EQ(pool.size(), pool.capacity());
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());
}


class nondefault_int_vector
{
    public:
        nondefault_int_vector() = delete;

        STDGPU_HOST_DEVICE
        nondefault_int_vector(const int x)
            : x(x)
        {

        }

        STDGPU_HOST_DEVICE
        operator int() const
        {
            return x;
        }

    private:
        int x;
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
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

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
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, pop_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    stdgpu::vector<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, pop_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_vector_const_type<T>(pool, 2.0f));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_back_vector_const_type<T>(pool));

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.empty());
    EXPECT_FALSE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, pop_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_vector<nondefault_int_vector>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_back_vector<nondefault_int_vector>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, push_back_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1 + N_remaining;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_vector<int>(pool));


    thrust::sort(stdgpu::device_begin(pool), stdgpu::device_end(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_vector, push_back_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_vector<int>(pool));


    thrust::sort(stdgpu::device_begin(pool), stdgpu::device_end(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_vector, push_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_vector<int>(pool));


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
}


TEST_F(stdgpu_vector, push_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_vector_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, push_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_vector<nondefault_int_vector>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, emplace_back_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1 + N_remaining;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_vector<int>(pool));


    thrust::sort(stdgpu::device_begin(pool), stdgpu::device_end(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_vector, emplace_back_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_back_vector<int>(pool));


    thrust::sort(stdgpu::device_begin(pool), stdgpu::device_end(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_vector, emplace_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_back_vector<int>(pool));


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
}


TEST_F(stdgpu_vector, emplace_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::vector<T> pool = stdgpu::vector<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_back_vector_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_vector, emplace_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::vector<nondefault_int_vector> pool = stdgpu::vector<nondefault_int_vector>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_back_vector<nondefault_int_vector>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::vector<nondefault_int_vector>::destroyDeviceObject(pool);
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

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_vector<nondefault_int_vector>(pool));

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
}


template <typename T>
struct simultaneous_push_back_and_pop_back_vector
{
    stdgpu::vector<T> pool;
    stdgpu::vector<T> pool_validation;

    simultaneous_push_back_and_pop_back_vector(stdgpu::vector<T> pool,
                                               stdgpu::vector<T> pool_validation)
        : pool(pool),
          pool_validation(pool_validation)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.push_back(x);

        thrust::pair<T, bool> popped = pool.pop_back();

        if (popped.second)
        {
            pool_validation.push_back(popped.first);
        }
    }
};

TEST_F(stdgpu_vector, simultaneous_push_back_and_pop_back)
{
    const stdgpu::index_t N = 100000;

    stdgpu::vector<int> pool            = stdgpu::vector<int>::createDeviceObject(N);
    stdgpu::vector<int> pool_validation = stdgpu::vector<int>::createDeviceObject(N);

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N + init),
                     simultaneous_push_back_and_pop_back_vector<int>(pool, pool_validation));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());


    thrust::sort(stdgpu::device_begin(pool_validation), stdgpu::device_end(pool_validation));

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    stdgpu::vector<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
}


struct at_non_const_vector
{
    stdgpu::vector<int> pool;

    at_non_const_vector(stdgpu::vector<int> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const int x)
    {
        pool.at(x) = x * x;
    }
};


TEST_F(stdgpu_vector, at_non_const)
{
    const stdgpu::index_t N = 100000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     at_non_const_vector(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i * i);
    }

    stdgpu::vector<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


struct access_operator_non_const_vector
{
    stdgpu::vector<int> pool;

    access_operator_non_const_vector(stdgpu::vector<int> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const int x)
    {
        pool[x] = x * x;
    }
};


TEST_F(stdgpu_vector, access_operator_non_const)
{
    const stdgpu::index_t N = 100000;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     access_operator_non_const_vector(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i * i);
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
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 100;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::vector<int> pool = stdgpu::vector<int>::createDeviceObject(N);

    fill_vector(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_vector<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), N_remaining);
    EXPECT_TRUE(pool.capacity() == N_remaining || pool.capacity() == N); // Capacity may have changed or not
    EXPECT_TRUE(pool.valid());

    stdgpu::vector<int>::destroyDeviceObject(pool);
}


