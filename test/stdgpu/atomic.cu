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
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <test_utils.h>
#include <stdgpu/atomic.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_atomic : public ::testing::Test
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

    __device__ void
    operator()(const T x)
    {
        atomicSub(value, x);
    }
};


TEST_F(stdgpu_atomic, unsigned_long_long_int_sub)
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


TEST_F(stdgpu_atomic, unsigned_long_long_int_sub_zero_pattern)
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


TEST_F(stdgpu_atomic, unsigned_long_long_int_sub_one_pattern)
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


TEST_F(stdgpu_atomic, float_sub)
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

    __device__ void
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

    __device__ void
    operator()(const float x)
    {
        atomicMax(value, x);
    }
};


TEST_F(stdgpu_atomic, float_min)
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


TEST_F(stdgpu_atomic, float_max)
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



template <typename T>
void
load_and_store()
{
    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    EXPECT_EQ(value.load(), T());

    value.store(T(42));

    EXPECT_EQ(value.load(), T(42));

    stdgpu::atomic<T>::destroyDeviceObject(value);
}



TEST_F(stdgpu_atomic, load_and_store_int)
{
    load_and_store<int>();
}

TEST_F(stdgpu_atomic, load_and_store_unsigned_int)
{
    load_and_store<unsigned int>();
}

TEST_F(stdgpu_atomic, load_and_store_unsigned_long_long_int)
{
    load_and_store<unsigned long long int>();
}


template <typename T>
struct sum_seqeuence
{
    stdgpu::atomic<T> value;

    sum_seqeuence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_add(x);
    }
};


template <typename T>
void
sequence_fetch_add()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     sum_seqeuence<T>(value));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_add_int)
{
    sequence_fetch_add<int>();
}

TEST_F(stdgpu_atomic, fetch_add_unsigned_int)
{
    sequence_fetch_add<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_add_unsigned_long_long_int)
{
    sequence_fetch_add<unsigned long long int>();
}


template <typename T>
struct desum_sequence
{
    stdgpu::atomic<T> value;

    desum_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_sub(x);
    }
};

template <typename T>
void
sequence_fetch_sub()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     sum_seqeuence<T>(value));

    ASSERT_EQ(value.load(), T(N * (N + 1) / 2));

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     desum_sequence<T>(value));

    EXPECT_EQ(value.load(), T(0));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_sub_int)
{
    sequence_fetch_sub<int>();
}

TEST_F(stdgpu_atomic, fetch_sub_unsigned_int)
{
    sequence_fetch_sub<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_sub_unsigned_long_long_int)
{
    sequence_fetch_sub<unsigned long long int>();
}


template <typename T>
bool
bit_set(const T value,
        const int bit_position)
{
    return (1 == ( (value >> bit_position) & 1));
}


template <typename T>
struct or_sequence
{
    stdgpu::atomic<T> value;

    or_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const stdgpu::index_t i)
    {
        T pattern = 1 << i;

        value.fetch_or(pattern);
    }
};


template <typename T>
void
sequence_fetch_or()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     or_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_or_int)
{
    sequence_fetch_or<int>();
}

TEST_F(stdgpu_atomic, fetch_or_unsigned_int)
{
    sequence_fetch_or<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_or_unsigned_long_long_int)
{
    sequence_fetch_or<unsigned long long int>();
}


template <typename T>
struct and_sequence
{
    stdgpu::atomic<T> value;
    T one_pattern;

    and_sequence(stdgpu::atomic<T> value,
                 T one_pattern)
        : value(value),
          one_pattern(one_pattern)
    {

    }

    __device__ void
    operator()(const stdgpu::index_t i)
    {
        T pattern = one_pattern - (1 << i);

        value.fetch_and(pattern);
    }
};


template <typename T>
void
sequence_fetch_and()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     or_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        ASSERT_TRUE(bit_set(value_pattern, i));
    }

    T one_pattern = value.load();   // We previously filled this with 1's

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     and_sequence<T>(value, one_pattern));

    value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_FALSE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_and_int)
{
    sequence_fetch_and<int>();
}

TEST_F(stdgpu_atomic, fetch_and_unsigned_int)
{
    sequence_fetch_and<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_and_unsigned_long_long_int)
{
    sequence_fetch_and<unsigned long long int>();
}


template <typename T>
struct xor_sequence
{
    stdgpu::atomic<T> value;

    xor_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const stdgpu::index_t i)
    {
        T pattern = 1 << i;

        value.fetch_xor(pattern);
    }
};


template <typename T>
void
sequence_fetch_xor()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     xor_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_xor_int)
{
    sequence_fetch_xor<int>();
}

TEST_F(stdgpu_atomic, fetch_xor_unsigned_int)
{
    sequence_fetch_xor<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_xor_unsigned_long_long_int)
{
    sequence_fetch_xor<unsigned long long int>();
}


template <typename T>
struct min_sequence
{
    stdgpu::atomic<T> value;

    min_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_min(x);
    }
};


template <typename T>
void
sequence_fetch_min()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(std::numeric_limits<T>::max());

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     min_sequence<T>(value));

    EXPECT_EQ(value.load(), T(1));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_min_int)
{
    sequence_fetch_min<int>();
}

TEST_F(stdgpu_atomic, fetch_min_unsigned_int)
{
    sequence_fetch_min<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_min_unsigned_long_long_int)
{
    sequence_fetch_min<unsigned long long int>();
}


template <typename T>
struct max_sequence
{
    stdgpu::atomic<T> value;

    max_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_max(x);
    }
};


template <typename T>
void
sequence_fetch_max()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(std::numeric_limits<T>::lowest());

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     max_sequence<T>(value));

    EXPECT_EQ(value.load(), T(N));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, fetch_max_int)
{
    sequence_fetch_max<int>();
}

TEST_F(stdgpu_atomic, fetch_max_unsigned_int)
{
    sequence_fetch_max<unsigned int>();
}

TEST_F(stdgpu_atomic, fetch_max_unsigned_long_long_int)
{
    sequence_fetch_max<unsigned long long int>();
}


template <typename T>
struct inc_mod_sequence
{
    stdgpu::atomic<T> value;

    inc_mod_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_inc_mod(x);
    }
};


template <typename T>
void
sequence_fetch_inc_mod()
{
    const stdgpu::index_t N = 50000;
    const stdgpu::index_t modulus_value = N / 10;
    T* sequence = createDeviceArray<T>(N, modulus_value);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(42);

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     inc_mod_sequence<T>(value));

    EXPECT_EQ(value.load(), T(42));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


/*TEST_F(stdgpu_atomic, fetch_inc_mod_int)
{
    sequence_fetch_inc_mod<int>();
}*/

TEST_F(stdgpu_atomic, fetch_inc_mod_unsigned_int)
{
    sequence_fetch_inc_mod<unsigned int>();
}

/*TEST_F(stdgpu_atomic, fetch_inc_mod_unsigned_long_long_int)
{
    sequence_fetch_inc_mod<unsigned long long int>();
}*/


template <typename T>
struct dec_mod_dequence
{
    stdgpu::atomic<T> value;

    dec_mod_dequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ void
    operator()(const T x)
    {
        value.fetch_dec_mod(x);
    }
};


template <typename T>
void
sequence_fetch_dec_mod()
{
    const stdgpu::index_t N = 50000;
    const stdgpu::index_t modulus_value = N / 10;
    T* sequence = createDeviceArray<T>(N, modulus_value);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(42);

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     dec_mod_dequence<T>(value));

    EXPECT_EQ(value.load(), T(42));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


/*TEST_F(stdgpu_atomic, fetch_dec_mod_int)
{
    sequence_fetch_dec_mod<int>();
}*/

TEST_F(stdgpu_atomic, fetch_dec_mod_unsigned_int)
{
    sequence_fetch_dec_mod<unsigned int>();
}

/*TEST_F(stdgpu_atomic, fetch_dec_mod_unsigned_long_long_int)
{
    sequence_fetch_dec_mod<unsigned long long int>();
}*/


template <typename T>
struct pre_inc_sequence
{
    stdgpu::atomic<T> value;

    pre_inc_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ T
    operator()()
    {
        return ++value;
    }
};


template <typename T>
void
sequence_pre_inc()
{
    const stdgpu::index_t N = 10000000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     pre_inc_sequence<T>(value));

    thrust::sort(stdgpu::device_begin(sequence), stdgpu::device_end(sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i + 1));
    }

    destroyDeviceArray<T>(sequence);
    destroyHostArray<T>(host_sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, pre_inc_operator_int)
{
    sequence_pre_inc<int>();
}

TEST_F(stdgpu_atomic, pre_inc_operator_unsigned_int)
{
    sequence_pre_inc<unsigned int>();
}

TEST_F(stdgpu_atomic, pre_inc_operator_unsigned_long_long_int)
{
    sequence_pre_inc<unsigned long long int>();
}


template <typename T>
struct post_inc_sequence
{
    stdgpu::atomic<T> value;

    post_inc_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ T
    operator()()
    {
        return value++;
    }
};


template <typename T>
void
sequence_post_inc()
{
    const stdgpu::index_t N = 10000000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     post_inc_sequence<T>(value));

    thrust::sort(stdgpu::device_begin(sequence), stdgpu::device_end(sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i));
    }

    destroyDeviceArray<T>(sequence);
    destroyHostArray<T>(host_sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, post_inc_operator_int)
{
    sequence_post_inc<int>();
}

TEST_F(stdgpu_atomic, post_inc_operator_unsigned_int)
{
    sequence_post_inc<unsigned int>();
}

TEST_F(stdgpu_atomic, post_inc_operator_unsigned_long_long_int)
{
    sequence_post_inc<unsigned long long int>();
}


template <typename T>
struct pre_dec_sequence
{
    stdgpu::atomic<T> value;

    pre_dec_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ T
    operator()()
    {
        return --value;
    }
};


template <typename T>
void
sequence_pre_dec()
{
    const stdgpu::index_t N = 10000000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     pre_inc_sequence<T>(value));

    ASSERT_EQ(value.load(), T(N));

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     pre_dec_sequence<T>(value));

    ASSERT_EQ(value.load(), T(0));

    thrust::sort(stdgpu::device_begin(sequence), stdgpu::device_end(sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i));
    }

    destroyDeviceArray<T>(sequence);
    destroyHostArray<T>(host_sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, pre_dec_operator_int)
{
    sequence_pre_dec<int>();
}

TEST_F(stdgpu_atomic, pre_dec_operator_unsigned_int)
{
    sequence_pre_dec<unsigned int>();
}

TEST_F(stdgpu_atomic, pre_dec_operator_unsigned_long_long_int)
{
    sequence_pre_dec<unsigned long long int>();
}


template <typename T>
struct post_dec_sequence
{
    stdgpu::atomic<T> value;

    post_dec_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    __device__ T
    operator()()
    {
        return value--;
    }
};


template <typename T>
void
sequence_post_dec()
{
    const stdgpu::index_t N = 10000000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     post_inc_sequence<T>(value));

    ASSERT_EQ(value.load(), T(N));

    thrust::generate(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     post_dec_sequence<T>(value));

    ASSERT_EQ(value.load(), T(0));

    thrust::sort(stdgpu::device_begin(sequence), stdgpu::device_end(sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i + 1));
    }

    destroyDeviceArray<T>(sequence);
    destroyHostArray<T>(host_sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, post_dec_operator_int)
{
    sequence_post_dec<int>();
}

TEST_F(stdgpu_atomic, post_dec_operator_unsigned_int)
{
    sequence_post_dec<unsigned int>();
}

TEST_F(stdgpu_atomic, post_dec_operator_unsigned_long_long_int)
{
    sequence_post_dec<unsigned long long int>();
}


