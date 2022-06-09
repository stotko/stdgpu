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
#include <limits>

#include <stdgpu/algorithm.h>
#include <stdgpu/atomic.cuh>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/numeric.h>
#include <test_memory_utils.h>
#include <test_utils.h>

class stdgpu_atomic : public ::testing::Test
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

template class atomic<unsigned int>;

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_add(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_sub(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_and(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_or(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_xor(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_min(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_max(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_inc_mod(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::fetch_dec_mod(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator++();

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator++(int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator--();

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator--(int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator+=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator-=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator&=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator|=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic<unsigned int>::operator^=(const unsigned int);

template class atomic_ref<unsigned int>;

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_add(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_sub(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_and(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_or(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_xor(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_min(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_max(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_inc_mod(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::fetch_dec_mod(const unsigned int, const memory_order order);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator++();

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator++(int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator--();

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator--(int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator+=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator-=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator&=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator|=(const unsigned int);

template STDGPU_DEVICE_ONLY unsigned int
atomic_ref<unsigned int>::operator^=(const unsigned int);

} // namespace stdgpu

template <typename T>
void
empty_container()
{
    stdgpu::atomic<T> empty_container;

    EXPECT_EQ(empty_container.load(), T());

    const T loaded = empty_container;
    EXPECT_EQ(loaded, T());

    const T new_value = static_cast<T>(42);
    empty_container.store(new_value);
    empty_container = new_value;
}

TEST_F(stdgpu_atomic, empty_container_int)
{
    empty_container<int>();
}

TEST_F(stdgpu_atomic, empty_container_unsigned_int)
{
    empty_container<unsigned int>();
}

TEST_F(stdgpu_atomic, empty_container_unsigned_long_long_int)
{
    empty_container<unsigned long long int>();
}

template <typename T>
void
is_lock_free()
{
    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    EXPECT_EQ(value.is_lock_free(), stdgpu::atomic_is_lock_free(&value));

    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, is_lock_free_int)
{
    is_lock_free<int>();
}

TEST_F(stdgpu_atomic, is_lock_free_unsigned_int)
{
    is_lock_free<unsigned int>();
}

TEST_F(stdgpu_atomic, is_lock_free_unsigned_long_long_int)
{
    is_lock_free<unsigned long long int>();
}

template <typename T>
class load_value
{
public:
    explicit load_value(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE T
    operator()() const
    {
        return _value.load();
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class store_value
{
public:
    explicit store_value(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const T x)
    {
        _value.store(x);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class load_value_nonmember
{
public:
    explicit load_value_nonmember(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE T
    operator()() const
    {
        return stdgpu::atomic_load(&_value);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class store_value_nonmember
{
public:
    explicit store_value_nonmember(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const T x)
    {
        stdgpu::atomic_store(&_value, x);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class load_value_nonmember_explicit
{
public:
    explicit load_value_nonmember_explicit(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE T
    operator()() const
    {
        return stdgpu::atomic_load_explicit(&_value, stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class store_value_nonmember_explicit
{
public:
    explicit store_value_nonmember_explicit(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const T x)
    {
        stdgpu::atomic_store_explicit(&_value, x, stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T, template <typename> class LoadFunction, template <typename> class StoreFunction>
void
load_and_store()
{
    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    const T new_value = static_cast<T>(42);

    EXPECT_EQ(LoadFunction<T>{ value }(), T());

    StoreFunction<T>{ value }(new_value);

    EXPECT_EQ(LoadFunction<T>{ value }(), new_value);

    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, load_and_store_int)
{
    load_and_store<int, load_value, store_value>();
}

TEST_F(stdgpu_atomic, load_and_store_unsigned_int)
{
    load_and_store<unsigned int, load_value, store_value>();
}

TEST_F(stdgpu_atomic, load_and_store_unsigned_long_long_int)
{
    load_and_store<unsigned long long int, load_value, store_value>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_int)
{
    load_and_store<int, load_value_nonmember, store_value_nonmember>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_unsigned_int)
{
    load_and_store<unsigned int, load_value_nonmember, store_value_nonmember>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_unsigned_long_long_int)
{
    load_and_store<unsigned long long int, load_value_nonmember, store_value_nonmember>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_explicit_int)
{
    load_and_store<int, load_value_nonmember_explicit, store_value_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_explicit_unsigned_int)
{
    load_and_store<unsigned int, load_value_nonmember_explicit, store_value_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, load_and_store_nonmember_explicit_unsigned_long_long_int)
{
    load_and_store<unsigned long long int, load_value_nonmember_explicit, store_value_nonmember_explicit>();
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
    exchange_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = _value.exchange(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class exchange_sequence_nonmember
{
public:
    exchange_sequence_nonmember(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = stdgpu::atomic_exchange(&_value, _sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class exchange_sequence_nonmember_explicit
{
public:
    exchange_sequence_nonmember_explicit(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = stdgpu::atomic_exchange_explicit(&_value, _sequence[i], stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T, template <typename> class Function>
void
sequence_exchange()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N - 1);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(N);

    stdgpu::for_each_index(stdgpu::execution::device, N - 1, Function<T>(value, sequence));

    T* host_sequence = createHostArray<T>(N);
    copyDevice2HostArray<T>(sequence, N - 1, host_sequence);
    host_sequence[N - 1] = value.load();

    std::sort(stdgpu::host_begin(host_sequence).get(), stdgpu::host_end(host_sequence).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i + 1));
    }

    destroyHostArray<T>(host_sequence);
    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, exchange_int)
{
    sequence_exchange<int, exchange_sequence>();
}

TEST_F(stdgpu_atomic, exchange_unsigned_int)
{
    sequence_exchange<unsigned int, exchange_sequence>();
}

TEST_F(stdgpu_atomic, exchange_unsigned_long_long_int)
{
    sequence_exchange<unsigned long long int, exchange_sequence>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_int)
{
    sequence_exchange<int, exchange_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_unsigned_int)
{
    sequence_exchange<unsigned int, exchange_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_unsigned_long_long_int)
{
    sequence_exchange<unsigned long long int, exchange_sequence>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_explicit_int)
{
    sequence_exchange<int, exchange_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_explicit_unsigned_int)
{
    sequence_exchange<unsigned int, exchange_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, exchange_nonmember_explicit_unsigned_long_long_int)
{
    sequence_exchange<unsigned long long int, exchange_sequence_nonmember_explicit>();
}

template <typename T>
class add_sequence_with_compare_exchange_weak
{
public:
    add_sequence_with_compare_exchange_weak(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T old = _value.load();
        while (!_value.compare_exchange_weak(old, old + _sequence[i]))
        {
            // Wait until exchanged
        }
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_sequence_with_compare_exchange_weak_nonmember
{
public:
    add_sequence_with_compare_exchange_weak_nonmember(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T old = stdgpu::atomic_load(&_value);
        while (!stdgpu::atomic_compare_exchange_weak(&_value, &old, old + _sequence[i]))
        {
            // Wait until exchanged
        }
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_sequence_with_compare_exchange_strong
{
public:
    add_sequence_with_compare_exchange_strong(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T old = _value.load();
        while (!_value.compare_exchange_strong(old, old + _sequence[i]))
        {
            // Wait until exchanged
        }
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_sequence_with_compare_exchange_strong_nonmember
{
public:
    add_sequence_with_compare_exchange_strong_nonmember(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T old = stdgpu::atomic_load(&_value);
        while (!stdgpu::atomic_compare_exchange_strong(&_value, &old, old + _sequence[i]))
        {
            // Wait until exchanged
        }
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T, template <typename> class Function>
void
sequence_compare_exchange_weak()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value, sequence));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

template <typename T, template <typename> class Function>
void
sequence_compare_exchange_strong()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value, sequence));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, compare_exchange_weak_int)
{
    sequence_compare_exchange_weak<int, add_sequence_with_compare_exchange_weak>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_unsigned_int)
{
    sequence_compare_exchange_weak<unsigned int, add_sequence_with_compare_exchange_weak>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_unsigned_long_long_int)
{
    sequence_compare_exchange_weak<unsigned long long int, add_sequence_with_compare_exchange_weak>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_nonmember_int)
{
    sequence_compare_exchange_weak<int, add_sequence_with_compare_exchange_weak_nonmember>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_nonmember_unsigned_int)
{
    sequence_compare_exchange_weak<unsigned int, add_sequence_with_compare_exchange_weak_nonmember>();
}

TEST_F(stdgpu_atomic, compare_exchange_weak_nonmember_unsigned_long_long_int)
{
    sequence_compare_exchange_weak<unsigned long long int, add_sequence_with_compare_exchange_weak_nonmember>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_int)
{
    sequence_compare_exchange_strong<int, add_sequence_with_compare_exchange_strong>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_unsigned_int)
{
    sequence_compare_exchange_strong<unsigned int, add_sequence_with_compare_exchange_strong>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_unsigned_long_long_int)
{
    sequence_compare_exchange_strong<unsigned long long int, add_sequence_with_compare_exchange_strong>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_nonmember_int)
{
    sequence_compare_exchange_weak<int, add_sequence_with_compare_exchange_strong_nonmember>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_nonmember_unsigned_int)
{
    sequence_compare_exchange_weak<unsigned int, add_sequence_with_compare_exchange_strong_nonmember>();
}

TEST_F(stdgpu_atomic, compare_exchange_strong_nonmember_unsigned_long_long_int)
{
    sequence_compare_exchange_weak<unsigned long long int, add_sequence_with_compare_exchange_strong_nonmember>();
}

template <typename T>
class add_sequence
{
public:
    add_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_add(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_sequence_nonmember
{
public:
    add_sequence_nonmember(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::atomic_fetch_add(&_value, _sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_sequence_nonmember_explicit
{
public:
    add_sequence_nonmember_explicit(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::atomic_fetch_add_explicit(&_value, _sequence[i], stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class add_equals_sequence
{
public:
    add_equals_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value += _sequence[i];
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T, template <typename> class Function>
void
sequence_fetch_add()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value, sequence));

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
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, add_equals_sequence<T>(value, sequence));

    EXPECT_EQ(value.load(), T(N * (N + 1) / 2));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fetch_add_int)
{
    sequence_fetch_add<int, add_sequence>();
}

TEST_F(stdgpu_atomic, fetch_add_unsigned_int)
{
    sequence_fetch_add<unsigned int, add_sequence>();
}

TEST_F(stdgpu_atomic, fetch_add_unsigned_long_long_int)
{
    sequence_fetch_add<unsigned long long int, add_sequence>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_int)
{
    sequence_fetch_add<int, add_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_unsigned_int)
{
    sequence_fetch_add<unsigned int, add_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_unsigned_long_long_int)
{
    sequence_fetch_add<unsigned long long int, add_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_explicit_int)
{
    sequence_fetch_add<int, add_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_explicit_unsigned_int)
{
    sequence_fetch_add<unsigned int, add_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_add_nonmember_explicit_unsigned_long_long_int)
{
    sequence_fetch_add<unsigned long long int, add_sequence_nonmember_explicit>();
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
    sub_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_sub(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class sub_sequence_nonmember
{
public:
    sub_sequence_nonmember(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::atomic_fetch_sub(&_value, _sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class sub_sequence_nonmember_explicit
{
public:
    sub_sequence_nonmember_explicit(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::atomic_fetch_sub_explicit(&_value, _sequence[i], stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
class sub_equals_sequence
{
public:
    sub_equals_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value -= _sequence[i];
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T, template <typename> class Function>
void
sequence_fetch_sub()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, add_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(N * (N + 1) / 2));

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value, sequence));

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
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, add_equals_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(N * (N + 1) / 2));

    stdgpu::for_each_index(stdgpu::execution::device, N, sub_equals_sequence<T>(value, sequence));

    EXPECT_EQ(value.load(), T(0));

    destroyDeviceArray<T>(sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fetch_sub_int)
{
    sequence_fetch_sub<int, sub_sequence>();
}

TEST_F(stdgpu_atomic, fetch_sub_unsigned_int)
{
    sequence_fetch_sub<unsigned int, sub_sequence>();
}

TEST_F(stdgpu_atomic, fetch_sub_unsigned_long_long_int)
{
    sequence_fetch_sub<unsigned long long int, sub_sequence>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_int)
{
    sequence_fetch_sub<int, sub_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_unsigned_int)
{
    sequence_fetch_sub<unsigned int, sub_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_unsigned_long_long_int)
{
    sequence_fetch_sub<unsigned long long int, sub_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_explicit_int)
{
    sequence_fetch_sub<int, sub_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_explicit_unsigned_int)
{
    sequence_fetch_sub<unsigned int, sub_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_sub_nonmember_explicit_unsigned_long_long_int)
{
    sequence_fetch_sub<unsigned long long int, sub_sequence_nonmember_explicit>();
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
bit_set(const T value, const stdgpu::index_t bit_position)
{
    return (1 == ((value >> bit_position) & 1)); // NOLINT(hicpp-signed-bitwise)
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
class or_sequence_nonmember
{
public:
    explicit or_sequence_nonmember(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_or(&_value, pattern);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class or_sequence_nonmember_explicit
{
public:
    explicit or_sequence_nonmember_explicit(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_or_explicit(&_value, pattern, stdgpu::memory_order_relaxed);
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

template <typename T, template <typename> class Function>
void
sequence_fetch_or()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value));

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

    stdgpu::for_each_index(stdgpu::execution::device, N, or_equals_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fetch_or_int)
{
    sequence_fetch_or<int, or_sequence>();
}

TEST_F(stdgpu_atomic, fetch_or_unsigned_int)
{
    sequence_fetch_or<unsigned int, or_sequence>();
}

TEST_F(stdgpu_atomic, fetch_or_unsigned_long_long_int)
{
    sequence_fetch_or<unsigned long long int, or_sequence>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_int)
{
    sequence_fetch_or<int, or_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_unsigned_int)
{
    sequence_fetch_or<unsigned int, or_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_unsigned_long_long_int)
{
    sequence_fetch_or<unsigned long long int, or_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_explicit_int)
{
    sequence_fetch_or<int, or_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_explicit_unsigned_int)
{
    sequence_fetch_or<unsigned int, or_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_or_nonmember_explicit_unsigned_long_long_int)
{
    sequence_fetch_or<unsigned long long int, or_sequence_nonmember_explicit>();
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
    and_sequence(const stdgpu::atomic<T>& value, T one_pattern)
      : _value(value)
      , _one_pattern(one_pattern)
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
    T* _sequence;
    T _one_pattern;
};

template <typename T>
class and_sequence_nonmember
{
public:
    and_sequence_nonmember(const stdgpu::atomic<T>& value, T one_pattern)
      : _value(value)
      , _one_pattern(one_pattern)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = _one_pattern - (static_cast<T>(1) << i); // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_and(&_value, pattern);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
    T _one_pattern;
};

template <typename T>
class and_sequence_nonmember_explicit
{
public:
    and_sequence_nonmember_explicit(const stdgpu::atomic<T>& value, T one_pattern)
      : _value(value)
      , _one_pattern(one_pattern)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = _one_pattern - (static_cast<T>(1) << i); // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_and_explicit(&_value, pattern, stdgpu::memory_order_relaxed);
    }

private:
    stdgpu::atomic<T> _value;
    T _one_pattern;
};

template <typename T>
class and_equals_sequence
{
public:
    and_equals_sequence(const stdgpu::atomic<T>& value, T one_pattern)
      : _value(value)
      , _one_pattern(one_pattern)
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

template <typename T, template <typename> class Function>
void
sequence_fetch_and()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, or_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        ASSERT_TRUE(bit_set(value_pattern, i));
    }

    T one_pattern = value.load(); // We previously filled this with 1's

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value, one_pattern));

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

    stdgpu::for_each_index(stdgpu::execution::device, N, or_equals_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        ASSERT_TRUE(bit_set(value_pattern, i));
    }

    T one_pattern = value.load(); // We previously filled this with 1's

    stdgpu::for_each_index(stdgpu::execution::device, N, and_equals_sequence<T>(value, one_pattern));

    value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_FALSE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fetch_and_int)
{
    sequence_fetch_and<int, and_sequence>();
}

TEST_F(stdgpu_atomic, fetch_and_unsigned_int)
{
    sequence_fetch_and<unsigned int, and_sequence>();
}

TEST_F(stdgpu_atomic, fetch_and_unsigned_long_long_int)
{
    sequence_fetch_and<unsigned long long int, and_sequence>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_int)
{
    sequence_fetch_and<int, and_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_unsigned_int)
{
    sequence_fetch_and<unsigned int, and_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_unsigned_long_long_int)
{
    sequence_fetch_and<unsigned long long int, and_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_explicit_int)
{
    sequence_fetch_and<int, and_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_explicit_unsigned_int)
{
    sequence_fetch_and<unsigned int, and_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_and_nonmember_explicit_unsigned_long_long_int)
{
    sequence_fetch_and<unsigned long long int, and_sequence_nonmember_explicit>();
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
class xor_sequence_nonmember
{
public:
    explicit xor_sequence_nonmember(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_xor(&_value, pattern);
    }

private:
    stdgpu::atomic<T> _value;
};

template <typename T>
class xor_sequence_nonmember_explicit
{
public:
    explicit xor_sequence_nonmember_explicit(const stdgpu::atomic<T>& value)
      : _value(value)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        T pattern = static_cast<T>(1) << i; // NOLINT(hicpp-signed-bitwise)

        stdgpu::atomic_fetch_xor_explicit(&_value, pattern, stdgpu::memory_order_relaxed);
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

template <typename T, template <typename> class Function>
void
sequence_fetch_xor()
{
    const stdgpu::index_t N = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, Function<T>(value));

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

    stdgpu::for_each_index(stdgpu::execution::device, N, xor_equals_sequence<T>(value));

    T value_pattern = value.load();
    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_TRUE(bit_set(value_pattern, i));
    }

    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fetch_xor_int)
{
    sequence_fetch_xor<int, xor_sequence>();
}

TEST_F(stdgpu_atomic, fetch_xor_unsigned_int)
{
    sequence_fetch_xor<unsigned int, xor_sequence>();
}

TEST_F(stdgpu_atomic, fetch_xor_unsigned_long_long_int)
{
    sequence_fetch_xor<unsigned long long int, xor_sequence>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_int)
{
    sequence_fetch_xor<int, xor_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_unsigned_int)
{
    sequence_fetch_xor<unsigned int, xor_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_unsigned_long_long_int)
{
    sequence_fetch_xor<unsigned long long int, xor_sequence_nonmember>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_explicit_int)
{
    sequence_fetch_xor<int, xor_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_explicit_unsigned_int)
{
    sequence_fetch_xor<unsigned int, xor_sequence_nonmember_explicit>();
}

TEST_F(stdgpu_atomic, fetch_xor_nonmember_explicit_unsigned_long_long_int)
{
    sequence_fetch_xor<unsigned long long int, xor_sequence_nonmember_explicit>();
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
    min_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_min(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_fetch_min()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(std::numeric_limits<T>::max());

    stdgpu::for_each_index(stdgpu::execution::device, N, min_sequence<T>(value, sequence));

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
    max_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_max(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_fetch_max()
{
    const stdgpu::index_t N = 40000;
    T* sequence = createDeviceArray<T>(N);
    stdgpu::iota(stdgpu::execution::device, stdgpu::device_begin(sequence), stdgpu::device_end(sequence), T(1));

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();
    value.store(std::numeric_limits<T>::lowest());

    stdgpu::for_each_index(stdgpu::execution::device, N, max_sequence<T>(value, sequence));

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
    inc_mod_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_inc_mod(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
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

    stdgpu::for_each_index(stdgpu::execution::device, N, inc_mod_sequence<T>(value, sequence));

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
    dec_mod_dequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _value.fetch_dec_mod(_sequence[i]);
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
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

    stdgpu::for_each_index(stdgpu::execution::device, N, dec_mod_dequence<T>(value, sequence));

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
    pre_inc_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = ++_value;
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_pre_inc()
{
    const stdgpu::index_t N = 100000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, pre_inc_sequence<T>(value, sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    std::sort(stdgpu::device_begin(host_sequence).get(), stdgpu::device_end(host_sequence).get());

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
    post_inc_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = _value++;
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_post_inc()
{
    const stdgpu::index_t N = 100000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, post_inc_sequence<T>(value, sequence));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    std::sort(stdgpu::device_begin(host_sequence).get(), stdgpu::device_end(host_sequence).get());

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
    pre_dec_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = --_value;
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_pre_dec()
{
    const stdgpu::index_t N = 100000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, pre_inc_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(N));

    stdgpu::for_each_index(stdgpu::execution::device, N, pre_dec_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(0));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    std::sort(stdgpu::device_begin(host_sequence).get(), stdgpu::device_end(host_sequence).get());

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
    post_dec_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _sequence[i] = _value--;
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_post_dec()
{
    const stdgpu::index_t N = 100000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, post_inc_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(N));

    stdgpu::for_each_index(stdgpu::execution::device, N, post_dec_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(0));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    std::sort(stdgpu::device_begin(host_sequence).get(), stdgpu::device_end(host_sequence).get());

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

template <typename T>
class fence_sequence
{
public:
    fence_sequence(const stdgpu::atomic<T>& value, T* sequence)
      : _value(value)
      , _sequence(sequence)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::atomic_thread_fence(stdgpu::memory_order_seq_cst);
        stdgpu::atomic_signal_fence(stdgpu::memory_order_seq_cst);

        T result = _value.fetch_add(1, stdgpu::memory_order_relaxed);

        stdgpu::atomic_thread_fence(stdgpu::memory_order_seq_cst);
        stdgpu::atomic_signal_fence(stdgpu::memory_order_seq_cst);

        _sequence[i] = result;
    }

private:
    stdgpu::atomic<T> _value;
    T* _sequence;
};

template <typename T>
void
sequence_fence()
{
    const stdgpu::index_t N = 100000;
    T* sequence = createDeviceArray<T>(N);

    stdgpu::atomic<T> value = stdgpu::atomic<T>::createDeviceObject();

    stdgpu::for_each_index(stdgpu::execution::device, N, fence_sequence<T>(value, sequence));

    ASSERT_EQ(value.load(), T(N));

    T* host_sequence = copyCreateDevice2HostArray<T>(sequence, N);

    std::sort(stdgpu::device_begin(host_sequence).get(), stdgpu::device_end(host_sequence).get());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_sequence[i], static_cast<T>(i));
    }

    destroyDeviceArray<T>(sequence);
    destroyHostArray<T>(host_sequence);
    stdgpu::atomic<T>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, fence_int)
{
    sequence_fence<int>();
}

TEST_F(stdgpu_atomic, fence_unsigned_int)
{
    sequence_fence<unsigned int>();
}

TEST_F(stdgpu_atomic, fence_unsigned_long_long_int)
{
    sequence_fence<unsigned long long int>();
}

TEST_F(stdgpu_atomic, get_allocator)
{
    const stdgpu::index_t N = 10000;

    stdgpu::atomic<int> value = stdgpu::atomic<int>::createDeviceObject();

    stdgpu::atomic<int>::allocator_type a = value.get_allocator();

    int* array = a.allocate(N);
    a.deallocate(array, N);

    stdgpu::atomic<int>::destroyDeviceObject(value);
}

TEST_F(stdgpu_atomic, custom_allocator)
{
    test_utils::get_allocator_statistics().reset();

    {
        const stdgpu::index_t N = 10000;

        using Allocator = test_utils::test_device_allocator<int>;
        Allocator a_orig;

        stdgpu::atomic<int, Allocator> value = stdgpu::atomic<int, Allocator>::createDeviceObject(a_orig);

        stdgpu::atomic<int, Allocator>::allocator_type a = value.get_allocator();

        int* array = a.allocate(N);
        a.deallocate(array, N);

        stdgpu::atomic<int, Allocator>::destroyDeviceObject(value);
    }

    // Account for potential but not guaranteed copy-ellision
    EXPECT_EQ(test_utils::get_allocator_statistics().default_constructions, 1);
    EXPECT_GE(test_utils::get_allocator_statistics().copy_constructions, 2);
    EXPECT_LE(test_utils::get_allocator_statistics().copy_constructions, 3);
    EXPECT_GE(test_utils::get_allocator_statistics().destructions, 3);
    EXPECT_LE(test_utils::get_allocator_statistics().destructions, 4);

    test_utils::get_allocator_statistics().reset();
}
