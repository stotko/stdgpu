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

#include <stdgpu/deque.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_deque : public ::testing::Test
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
struct pop_back_deque
{
    stdgpu::deque<T> pool;

    pop_back_deque(stdgpu::deque<T> pool)
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
struct pop_back_deque_const_type
{
    stdgpu::deque<Pair> pool;

    pop_back_deque_const_type(stdgpu::deque<Pair> pool)
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
struct push_back_deque
{
    stdgpu::deque<T> pool;

    push_back_deque(stdgpu::deque<T> pool)
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
struct push_back_deque_const_type
{
    stdgpu::deque<Pair> pool;
    typename Pair::second_type second;

    push_back_deque_const_type(stdgpu::deque<Pair> pool,
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
struct emplace_back_deque
{
    stdgpu::deque<T> pool;

    emplace_back_deque(stdgpu::deque<T> pool)
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
struct emplace_back_deque_const_type
{
    stdgpu::deque<Pair> pool;
    typename Pair::second_type second;

    emplace_back_deque_const_type(stdgpu::deque<Pair> pool,
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
fill_deque(stdgpu::deque<int> pool)
{
    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(pool.capacity() + init),
                     push_back_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), pool.capacity());
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());
}


class nondefault_int_deque
{
    public:
        nondefault_int_deque() = delete;

        STDGPU_HOST_DEVICE
        nondefault_int_deque(const int x)
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


TEST_F(stdgpu_deque, create_destroy_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_back_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N_remaining; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, pop_back_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque_const_type<T>(pool, 2.0f));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_back_deque_const_type<T>(pool));

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.empty());
    EXPECT_FALSE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, push_back_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1 + N_remaining;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_deque<int>(pool));


    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_back_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_deque<int>(pool));


    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_deque<int>(pool));


    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        // Only test if all numbers are inside range since N_push - N_pop threads had no chance to insert their numbers
        EXPECT_GE(host_numbers[i], 1);
        EXPECT_LE(host_numbers[i], static_cast<int>(N_push));
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, push_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, emplace_back_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1 + N_remaining;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_back_deque<int>(pool));


    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_back_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_back_deque<int>(pool));


    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_back_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_back_deque<int>(pool));


    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        // Only test if all numbers are inside range since N_push - N_pop threads had no chance to insert their numbers
        EXPECT_GE(host_numbers[i], 1);
        EXPECT_LE(host_numbers[i], static_cast<int>(N_push));
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_back_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_back_deque_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, emplace_back_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


template <typename T>
struct pop_front_deque
{
    stdgpu::deque<T> pool;

    pop_front_deque(stdgpu::deque<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(STDGPU_MAYBE_UNUSED const T x)
    {
        pool.pop_front();
    }
};

template <typename Pair>
struct pop_front_deque_const_type
{
    stdgpu::deque<Pair> pool;

    pop_front_deque_const_type(stdgpu::deque<Pair> pool)
        : pool(pool)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(STDGPU_MAYBE_UNUSED const typename Pair::first_type& first)
    {
        pool.pop_front();
    }
};


template <typename T>
struct push_front_deque
{
    stdgpu::deque<T> pool;

    push_front_deque(stdgpu::deque<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.push_front(x);
    }
};

template <typename Pair>
struct push_front_deque_const_type
{
    stdgpu::deque<Pair> pool;
    typename Pair::second_type second;

    push_front_deque_const_type(stdgpu::deque<Pair> pool,
                                const typename Pair::second_type& second)
        : pool(pool),
          second(second)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(const typename Pair::first_type& first)
    {
        pool.push_front(thrust::make_pair(first, second));
    }
};


template <typename T>
struct emplace_front_deque
{
    stdgpu::deque<T> pool;

    emplace_front_deque(stdgpu::deque<T> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.emplace_front(x);
    }
};

template <typename Pair>
struct emplace_front_deque_const_type
{
    stdgpu::deque<Pair> pool;
    typename Pair::second_type second;

    emplace_front_deque_const_type(stdgpu::deque<Pair> pool,
                                   const typename Pair::second_type& second)
        : pool(pool),
          second(second)
    {

    }

    inline STDGPU_HOST_DEVICE void
    operator()(const typename Pair::first_type& first)
    {
        pool.emplace_front(first, second);
    }
};


TEST_F(stdgpu_deque, pop_front_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_FALSE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = N_pop; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, pop_front_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_front_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    // pool may be valid or invalid depending on the thread scheduling

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_front_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque_const_type<T>(pool, 2.0f));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_front_deque_const_type<T>(pool));

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.empty());
    EXPECT_FALSE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, pop_front_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     pop_front_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, push_front_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_front_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_front_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_front_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_front_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_front_deque<int>(pool));


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

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_front_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_front_deque_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, push_front_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_front_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, emplace_front_some)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 1000;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_front_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_front_all)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_front_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_front_too_many)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N;
    const stdgpu::index_t N_push       = N_pop + 1;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     emplace_front_deque<int>(pool));


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

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, emplace_front_const_type)
{
    using T = thrust::pair<int, const float>;

    const stdgpu::index_t N = 10000;

    stdgpu::deque<T> pool = stdgpu::deque<T>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_front_deque_const_type<T>(pool, 2.0f));

    EXPECT_EQ(pool.size(), N);
    EXPECT_FALSE(pool.empty());
    EXPECT_TRUE(pool.full());
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<T>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, emplace_front_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_front_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, push_back_circular)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N * 1 / 3;
    const stdgpu::index_t N_push       = 2 * N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = N - N_pop + 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_back_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1 + N_pop);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, push_front_circular)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = N * 1 / 3;
    const stdgpu::index_t N_push       = 2 * N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_front_deque<int>(pool));

    const stdgpu::index_t init = N - N_pop + 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N_push + init),
                     push_front_deque<int>(pool));

    thrust::sort(stdgpu::make_device(pool.data()), stdgpu::make_device(pool.data() + pool.size()));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1 + N_pop);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, clear)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);


    pool.clear();


    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, clear_nondefault_type)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<nondefault_int_deque> pool = stdgpu::deque<nondefault_int_deque>::createDeviceObject(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     push_back_deque<nondefault_int_deque>(pool));

    ASSERT_EQ(pool.size(), N);
    ASSERT_FALSE(pool.empty());
    ASSERT_TRUE(pool.full());
    ASSERT_TRUE(pool.valid());

    pool.clear();

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    stdgpu::deque<nondefault_int_deque>::destroyDeviceObject(pool);
}


template <typename T>
struct simultaneous_push_back_and_pop_back_deque
{
    stdgpu::deque<T> pool;
    stdgpu::deque<T> pool_validation;

    simultaneous_push_back_and_pop_back_deque(stdgpu::deque<T> pool,
                                              stdgpu::deque<T> pool_validation)
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


TEST_F(stdgpu_deque, simultaneous_push_back_and_pop_back)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool            = stdgpu::deque<int>::createDeviceObject(N);
    stdgpu::deque<int> pool_validation = stdgpu::deque<int>::createDeviceObject(N);

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N + init),
                     simultaneous_push_back_and_pop_back_deque<int>(pool, pool_validation));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::sort(stdgpu::make_device(pool_validation.data()), stdgpu::make_device(pool_validation.data() + pool_validation.size()));

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    stdgpu::deque<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
}


template <typename T>
struct simultaneous_push_front_and_pop_front_deque
{
    stdgpu::deque<T> pool;
    stdgpu::deque<T> pool_validation;

    simultaneous_push_front_and_pop_front_deque(stdgpu::deque<T> pool,
                                                stdgpu::deque<T> pool_validation)
        : pool(pool),
          pool_validation(pool_validation)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.push_front(x);

        thrust::pair<T, bool> popped = pool.pop_front();

        if (popped.second)
        {
            pool_validation.push_front(popped.first);
        }
    }
};


TEST_F(stdgpu_deque, simultaneous_push_front_and_pop_front)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool            = stdgpu::deque<int>::createDeviceObject(N);
    stdgpu::deque<int> pool_validation = stdgpu::deque<int>::createDeviceObject(N);

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N + init),
                     simultaneous_push_front_and_pop_front_deque<int>(pool, pool_validation));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::sort(stdgpu::make_device(pool_validation.data()), stdgpu::make_device(pool_validation.data() + pool_validation.size()));

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    stdgpu::deque<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
}


template <typename T>
struct simultaneous_push_front_and_pop_back_deque
{
    stdgpu::deque<T> pool;
    stdgpu::deque<T> pool_validation;

    simultaneous_push_front_and_pop_back_deque(stdgpu::deque<T> pool,
                                               stdgpu::deque<T> pool_validation)
        : pool(pool),
          pool_validation(pool_validation)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        pool.push_front(x);

        thrust::pair<T, bool> popped = pool.pop_back();

        if (popped.second)
        {
            pool_validation.push_front(popped.first);
        }
    }
};


TEST_F(stdgpu_deque, simultaneous_push_front_and_pop_back)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool            = stdgpu::deque<int>::createDeviceObject(N);
    stdgpu::deque<int> pool_validation = stdgpu::deque<int>::createDeviceObject(N);

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N + init),
                     simultaneous_push_front_and_pop_back_deque<int>(pool, pool_validation));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::sort(stdgpu::make_device(pool_validation.data()), stdgpu::make_device(pool_validation.data() + pool_validation.size()));

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    stdgpu::deque<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
}


template <typename T>
struct simultaneous_push_back_and_pop_front_deque
{
    stdgpu::deque<T> pool;
    stdgpu::deque<T> pool_validation;

    simultaneous_push_back_and_pop_front_deque(stdgpu::deque<T> pool,
                                               stdgpu::deque<T> pool_validation)
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


TEST_F(stdgpu_deque, simultaneous_push_back_and_pop_front)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool            = stdgpu::deque<int>::createDeviceObject(N);
    stdgpu::deque<int> pool_validation = stdgpu::deque<int>::createDeviceObject(N);

    const stdgpu::index_t init = 1;
    thrust::for_each(thrust::counting_iterator<int>(init), thrust::counting_iterator<int>(N + init),
                     simultaneous_push_back_and_pop_front_deque<int>(pool, pool_validation));

    ASSERT_EQ(pool.size(), 0);
    ASSERT_TRUE(pool.empty());
    ASSERT_FALSE(pool.full());
    ASSERT_TRUE(pool.valid());

    thrust::sort(stdgpu::make_device(pool_validation.data()), stdgpu::make_device(pool_validation.data() + pool_validation.size()));

    ASSERT_EQ(pool_validation.size(), N);
    ASSERT_FALSE(pool_validation.empty());
    ASSERT_TRUE(pool_validation.full());
    ASSERT_TRUE(pool_validation.valid());

    int* host_numbers = copyCreateDevice2HostArray(pool_validation.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i + 1);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    stdgpu::deque<int>::destroyDeviceObject(pool_validation);
    destroyHostArray<int>(host_numbers);
}


struct at_non_const_deque
{
    stdgpu::deque<int> pool;

    at_non_const_deque(stdgpu::deque<int> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const int x)
    {
        pool.at(x) = x * x;
    }
};


TEST_F(stdgpu_deque, at_non_const)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     at_non_const_deque(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i * i);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


struct access_operator_non_const_deque
{
    stdgpu::deque<int> pool;

    access_operator_non_const_deque(stdgpu::deque<int> pool)
        : pool(pool)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const int x)
    {
        pool[x] = x * x;
    }
};


TEST_F(stdgpu_deque, access_operator_non_const)
{
    const stdgpu::index_t N = 100000;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     access_operator_non_const_deque(pool));

    int* host_numbers = copyCreateDevice2HostArray(pool.data(), N);
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_numbers[i], i * i);
    }

    stdgpu::deque<int>::destroyDeviceObject(pool);
    destroyHostArray<int>(host_numbers);
}


TEST_F(stdgpu_deque, shrink_to_fit_empty)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    ASSERT_EQ(pool.size(), 0);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), 0);
    EXPECT_TRUE(pool.capacity() == 0 || pool.capacity() == N); // Capacity may have changed or not
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, shrink_to_fit_full)
{
    const stdgpu::index_t N = 10000;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    ASSERT_EQ(pool.size(), N);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), N);
    EXPECT_EQ(pool.capacity(), N);
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


TEST_F(stdgpu_deque, shrink_to_fit)
{
    const stdgpu::index_t N            = 10000;
    const stdgpu::index_t N_pop        = 100;
    const stdgpu::index_t N_remaining  = N - N_pop;

    stdgpu::deque<int> pool = stdgpu::deque<int>::createDeviceObject(N);

    fill_deque(pool);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N_pop),
                     pop_back_deque<int>(pool));

    ASSERT_EQ(pool.size(), N_remaining);
    ASSERT_EQ(pool.capacity(), N);
    ASSERT_TRUE(pool.valid());

    pool.shrink_to_fit();

    EXPECT_EQ(pool.size(), N_remaining);
    EXPECT_TRUE(pool.capacity() == N_remaining || pool.capacity() == N); // Capacity may have changed or not
    EXPECT_TRUE(pool.valid());

    stdgpu::deque<int>::destroyDeviceObject(pool);
}


