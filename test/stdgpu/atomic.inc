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
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <test_utils.h>
#include <stdgpu/atomic.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_atomic : public ::testing::Test
{
    protected:
        // Called before each test
        void SetUp() override
        {

        }

        // Called after each test
        void TearDown() override
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
    const T new_value = static_cast<T>(42);

    EXPECT_EQ(value.load(), T());

    value.store(new_value);

    EXPECT_EQ(value.load(), new_value);

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
void
operator_load_and_store()
{
    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    const T new_value = static_cast<T>(42);

    EXPECT_EQ(value, T());

    value = new_value;

    EXPECT_EQ(value, new_value);

    stdgpu::atomic<T>::destroyDeviceObject(value);
}



TEST_F(stdgpu_atomic, operator_load_and_store_int)
{
    operator_load_and_store<int>();
}

TEST_F(stdgpu_atomic, operator_load_and_store_unsigned_int)
{
    operator_load_and_store<unsigned int>();
}

TEST_F(stdgpu_atomic, operator_load_and_store_unsigned_long_long_int)
{
    operator_load_and_store<unsigned long long int>();
}


template <typename T>
class exchange_sequence
{
    public:
        explicit exchange_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(T& x)
        {
            x = _value.exchange(x);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
void
sequence_exchange()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N - 1);
    thrust::sequence(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(N);

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     exchange_sequence<T>(value));

    T sum = value.load()
          + thrust::reduce(stdgpu::device_cbegin(sequence), stdgpu::device_cend(sequence),
                           T(0),
                           thrust::plus<T>());

    EXPECT_EQ(sum, T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}


TEST_F(stdgpu_atomic, exchange_int)
{
    sequence_exchange<int>();
}

TEST_F(stdgpu_atomic, exchange_unsigned_int)
{
    sequence_exchange<unsigned int>();
}

TEST_F(stdgpu_atomic, exchange_unsigned_long_long_int)
{
    sequence_exchange<unsigned long long int>();
}


template <typename T>
class add_sequence_with_compare_exchange_weak
{
    public:
        explicit add_sequence_with_compare_exchange_weak(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            T old = _value.load();
            while (!_value.compare_exchange_weak(old, old + x))
            {
                // Wait until exchanged
            }
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
class add_sequence_with_compare_exchange_strong
{
    public:
        explicit add_sequence_with_compare_exchange_strong(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            T old = _value.load();
            while (!_value.compare_exchange_strong(old, old + x))
            {
                // Wait until exchanged
            }
        }

    private:
        stdgpu::atomic<T> _value;
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
class add_sequence
{
    public:
        explicit add_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_add(x);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
class add_equals_sequence
{
    public:
        explicit add_equals_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value += x;
        }

    private:
        stdgpu::atomic<T> _value;
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
class sub_sequence
{
    public:
        explicit sub_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_sub(x);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
class sub_equals_sequence
{
    public:
        explicit sub_equals_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value -= x;
        }

    private:
        stdgpu::atomic<T> _value;
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
        const stdgpu::index_t bit_position)
{
    return (1 == ( (value >> bit_position) & 1)); // NOLINT(hicpp-signed-bitwise)
}


template <typename T>
class or_sequence
{
    public:
        explicit or_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

            _value.fetch_or(pattern);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
class or_equals_sequence
{
    public:
        explicit or_equals_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

            _value |= pattern;
        }

    private:
        stdgpu::atomic<T> _value;
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
class and_sequence
{
    public:
        and_sequence(const stdgpu::atomic<T>& value,
                     T one_pattern)
            : _value(value),
              _one_pattern(one_pattern)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = _one_pattern - (static_cast<T>(1) << i); // NOLINT(hicpp-signed-bitwise)

            _value.fetch_and(pattern);
        }

    private:
        stdgpu::atomic<T> _value;
        T _one_pattern;
};


template <typename T>
class and_equals_sequence
{
    public:
        and_equals_sequence(const stdgpu::atomic<T>& value,
                            T one_pattern)
            : _value(value),
              _one_pattern(one_pattern)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = _one_pattern - (static_cast<T>(1) << i); // NOLINT(hicpp-signed-bitwise)

            _value &= pattern;
        }

    private:
        stdgpu::atomic<T> _value;
        T _one_pattern;
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
class xor_sequence
{
    public:
        explicit xor_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

            _value.fetch_xor(pattern);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
class xor_equals_sequence
{
    public:
        explicit xor_equals_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const stdgpu::index_t i)
        {
            T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

            _value ^= pattern;
        }

    private:
        stdgpu::atomic<T> _value;
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
class min_sequence
{
    public:
        explicit min_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_min(x);
        }

    private:
        stdgpu::atomic<T> _value;
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
class max_sequence
{
    public:
        explicit max_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_max(x);
        }

    private:
        stdgpu::atomic<T> _value;
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
class inc_mod_sequence
{
    public:
        explicit inc_mod_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_inc_mod(x);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
void
sequence_fetch_inc_mod()
{
    const stdgpu::index_t N = 50000;
    const stdgpu::index_t modulus_value = N / 10;
    T* sequence = createDeviceArray<T>(N, modulus_value);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    const T new_value = static_cast<T>(42);
    value.store(new_value);

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     inc_mod_sequence<T>(value));

    EXPECT_EQ(value.load(), new_value);

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
class dec_mod_dequence
{
    public:
        explicit dec_mod_dequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY void
        operator()(const T x)
        {
            _value.fetch_dec_mod(x);
        }

    private:
        stdgpu::atomic<T> _value;
};


template <typename T>
void
sequence_fetch_dec_mod()
{
    const stdgpu::index_t N = 50000;
    const stdgpu::index_t modulus_value = N / 10;
    T* sequence = createDeviceArray<T>(N, modulus_value);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    const T new_value = static_cast<T>(42);
    value.store(new_value);

    thrust::for_each(stdgpu::device_begin(sequence), stdgpu::device_end(sequence),
                     dec_mod_dequence<T>(value));

    EXPECT_EQ(value.load(), new_value);

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
class pre_inc_sequence
{
    public:
        explicit pre_inc_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY T
        operator()()
        {
            return ++_value;
        }

    private:
        stdgpu::atomic<T> _value;
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
class post_inc_sequence
{
    public:
        explicit post_inc_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY T
        operator()()
        {
            return _value++;
        }

    private:
        stdgpu::atomic<T> _value;
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
class pre_dec_sequence
{
    public:
        explicit pre_dec_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY T
        operator()()
        {
            return --_value;
        }

    private:
        stdgpu::atomic<T> _value;
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
class post_dec_sequence
{
    public:
        explicit post_dec_sequence(const stdgpu::atomic<T>& value)
            : _value(value)
        {

        }

        STDGPU_DEVICE_ONLY T
        operator()()
        {
            return _value--;
        }

    private:
        stdgpu::atomic<T> _value;
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


