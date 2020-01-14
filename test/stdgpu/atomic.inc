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
#include <thrust/sequence.h>
#include <thrust/sort.h>

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


// Explicit template instantiations
namespace stdgpu
{

template
class atomic<unsigned int>;

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_add<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_sub<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_and<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_or<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_xor<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_min<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_max<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_inc_mod<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_dec_mod<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator++<unsigned int, void>();

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator++<unsigned int, void>(int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator--<unsigned int, void>();

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator--<unsigned int, void>(int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator+=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator-=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator&=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator|=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator^=<unsigned int, void>(const unsigned int);

template
class atomic_ref<unsigned int>;

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_add<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_sub<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_and<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_or<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_xor<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_min<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_max<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_inc_mod<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_dec_mod<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator++<unsigned int, void>();

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator++<unsigned int, void>(int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator--<unsigned int, void>();

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator--<unsigned int, void>(int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator+=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator-=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator&=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator|=<unsigned int, void>(const unsigned int);

template
STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator^=<unsigned int, void>(const unsigned int);

} // namespace stdgpu


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
struct add_sequence_with_compare_exchange_weak
{
    stdgpu::atomic<T> value;

    add_sequence_with_compare_exchange_weak(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        T old = value.load();
        while (!value.compare_exchange_weak(old, old + x))
        ;   // Wait until exchanged
    }
};


template <typename T>
struct add_sequence_with_compare_exchange_strong
{
    stdgpu::atomic<T> value;

    add_sequence_with_compare_exchange_strong(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        T old = value.load();
        while (!value.compare_exchange_strong(old, old + x))
        ;   // Wait until exchanged
    }
};


template <typename T>
void
sequence_compare_exchange_weak()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     add_sequence_with_compare_exchange_weak<T>(value));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


template <typename T>
void
sequence_compare_exchange_strong()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     add_sequence_with_compare_exchange_strong<T>(value));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, compare_exchange_weak_int)
{
    sequence_compare_exchange_weak<int>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_unsigned_int)
{
    sequence_compare_exchange_weak<unsigned int>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_unsigned_long_long_int)
{
    sequence_compare_exchange_weak<unsigned long long int>();
}


TEST_F(stdgpu_atomic, compare_exchange_strong_int)
{
    sequence_compare_exchange_strong<int>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_unsigned_int)
{
    sequence_compare_exchange_strong<unsigned int>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_unsigned_long_long_int)
{
    sequence_compare_exchange_strong<unsigned long long int>();
}


template <typename T>
struct add_sequence
{
    stdgpu::atomic<T> value;

    add_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        value.fetch_add(x);
    }
};


template <typename T>
struct add_equals_sequence
{
    stdgpu::atomic<T> value;

    add_equals_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        value += x;
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
                     add_sequence<T>(value));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


template <typename T>
void
sequence_operator_add_equals()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     add_equals_sequence<T>(value));

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


TEST_F(stdgpu_atomic, operator_add_equals_int)
{
    sequence_operator_add_equals<int>();
}

TEST_F(stdgpu_atomic, operator_add_equals_unsigned_int)
{
    sequence_operator_add_equals<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_add_equals_unsigned_long_long_int)
{
    sequence_operator_add_equals<unsigned long long int>();
}


template <typename T>
struct sub_sequence
{
    stdgpu::atomic<T> value;

    sub_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        value.fetch_sub(x);
    }
};


template <typename T>
struct sub_equals_sequence
{
    stdgpu::atomic<T> value;

    sub_equals_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const T x)
    {
        value -= x;
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
                     add_sequence<T>(value));

    ASSERT_EQ(value.load(), T(N * (N + 1) / 2));

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     sub_sequence<T>(value));

    EXPECT_EQ(value.load(), T(0));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


template <typename T>
void
sequence_operator_sub_equals()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     add_equals_sequence<T>(value));

    ASSERT_EQ(value.load(), T(N * (N + 1) / 2));

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     sub_equals_sequence<T>(value));

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


TEST_F(stdgpu_atomic, operator_sub_equals_int)
{
    sequence_operator_sub_equals<int>();
}

TEST_F(stdgpu_atomic, operator_sub_equals_unsigned_int)
{
    sequence_operator_sub_equals<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_sub_equals_unsigned_long_long_int)
{
    sequence_operator_sub_equals<unsigned long long int>();
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

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i;

        value.fetch_or(pattern);
    }
};


template <typename T>
struct or_equals_sequence
{
    stdgpu::atomic<T> value;

    or_equals_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i;

        value |= pattern;
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


template <typename T>
void
sequence_operator_or_equals()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     or_equals_sequence<T>(value));

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


TEST_F(stdgpu_atomic, operator_or_equals_int)
{
    sequence_operator_or_equals<int>();
}

TEST_F(stdgpu_atomic, operator_or_equals_unsigned_int)
{
    sequence_operator_or_equals<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_or_equals_unsigned_long_long_int)
{
    sequence_operator_or_equals<unsigned long long int>();
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

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = one_pattern - (static_cast<T>(1) << i);

        value.fetch_and(pattern);
    }
};


template <typename T>
struct and_equals_sequence
{
    stdgpu::atomic<T> value;
    T one_pattern;

    and_equals_sequence(stdgpu::atomic<T> value,
                        T one_pattern)
        : value(value),
          one_pattern(one_pattern)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = one_pattern - (static_cast<T>(1) << i);

        value &= pattern;
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


template <typename T>
void
sequence_operator_and_equals()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     or_equals_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        ASSERT_TRUE(bit_set(value_pattern, i));
    }

    T one_pattern = value.load();   // We previously filled this with 1's

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     and_equals_sequence<T>(value, one_pattern));

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


TEST_F(stdgpu_atomic, operator_and_equals_int)
{
    sequence_operator_and_equals<int>();
}

TEST_F(stdgpu_atomic, operator_and_equals_unsigned_int)
{
    sequence_operator_and_equals<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_and_equals_unsigned_long_long_int)
{
    sequence_operator_and_equals<unsigned long long int>();
}


template <typename T>
struct xor_sequence
{
    stdgpu::atomic<T> value;

    xor_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i;

        value.fetch_xor(pattern);
    }
};


template <typename T>
struct xor_equals_sequence
{
    stdgpu::atomic<T> value;

    xor_equals_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i;

        value ^= pattern;
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


template <typename T>
void
sequence_operator_xor_equals()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     xor_equals_sequence<T>(value));

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


TEST_F(stdgpu_atomic, operator_xor_equals_int)
{
    sequence_operator_xor_equals<int>();
}

TEST_F(stdgpu_atomic, operator_xor_equals_unsigned_int)
{
    sequence_operator_xor_equals<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_xor_equals_unsigned_long_long_int)
{
    sequence_operator_xor_equals<unsigned long long int>();
}


template <typename T>
struct min_sequence
{
    stdgpu::atomic<T> value;

    min_sequence(stdgpu::atomic<T> value)
        : value(value)
    {

    }

    STDGPU_DEVICE_ONLY void
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

    STDGPU_DEVICE_ONLY void
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

    STDGPU_DEVICE_ONLY void
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

    STDGPU_DEVICE_ONLY void
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

    STDGPU_DEVICE_ONLY T
    operator()()
    {
        return ++value;
    }
};


template <typename T>
void
sequence_pre_inc()
{
    const stdgpu::index_t N = 100000;
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

    STDGPU_DEVICE_ONLY T
    operator()()
    {
        return value++;
    }
};


template <typename T>
void
sequence_post_inc()
{
    const stdgpu::index_t N = 100000;
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

    STDGPU_DEVICE_ONLY T
    operator()()
    {
        return --value;
    }
};


template <typename T>
void
sequence_pre_dec()
{
    const stdgpu::index_t N = 100000;
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

    STDGPU_DEVICE_ONLY T
    operator()()
    {
        return value--;
    }
};


template <typename T>
void
sequence_post_dec()
{
    const stdgpu::index_t N = 100000;
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


