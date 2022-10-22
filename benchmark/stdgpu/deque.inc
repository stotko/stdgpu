/*
 *  Copyright 2022 Patrick Stotko
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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <limits>
#include <random>

#include <benchmark_utils.h>
#include <stdgpu/algorithm.h>
#include <stdgpu/deque.cuh>
#include <stdgpu/memory.h>

namespace
{
int*
create_values(const stdgpu::index_t N)
{
    // Generate true random numbers
    size_t seed = benchmark_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());

    int* host_values = createHostArray<int>(N);

    std::generate(host_values, host_values + N, [&dist, &rng]() { return dist(rng); });

    int* values = copyCreateHost2DeviceArray<int>(host_values, N);
    destroyHostArray<int>(host_values);

    return values;
}

template <typename T>
class push_back_deque
{
public:
    push_back_deque(const stdgpu::deque<T>& pool, T* values)
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
    stdgpu::deque<T> _pool;
    T* _values;
};

template <typename T>
class push_front_deque
{
public:
    push_front_deque(const stdgpu::deque<T>& pool, T* values)
      : _pool(pool)
      , _values(values)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _pool.push_front(_values[i]);
    }

private:
    stdgpu::deque<T> _pool;
    T* _values;
};

template <typename T>
class pop_back_deque
{
public:
    explicit pop_back_deque(const stdgpu::deque<T>& pool)
      : _pool(pool)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        _pool.pop_back();
    }

private:
    stdgpu::deque<T> _pool;
};

template <typename T>
class pop_front_deque
{
public:
    explicit pop_front_deque(const stdgpu::deque<T>& pool)
      : _pool(pool)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        _pool.pop_front();
    }

private:
    stdgpu::deque<T> _pool;
};
} // namespace

void
stdgpu_deque_push_back(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);
    int* values = create_values(deque_size);

    stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));
    d.clear();

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));

        state.PauseTiming();
        d.clear();
        state.ResumeTiming();
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
    destroyDeviceArray<int>(values);
}

void
stdgpu_deque_push_front(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);
    int* values = create_values(deque_size);

    stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_front_deque<int>(d, values));
    d.clear();

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_front_deque<int>(d, values));

        state.PauseTiming();
        d.clear();
        state.ResumeTiming();
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
    destroyDeviceArray<int>(values);
}

void
stdgpu_deque_pop_back(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);
    int* values = create_values(deque_size);

    stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, pop_back_deque<int>(d));

        state.PauseTiming();
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));
        state.ResumeTiming();
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
    destroyDeviceArray<int>(values);
}

void
stdgpu_deque_pop_front(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);
    int* values = create_values(deque_size);

    stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, pop_front_deque<int>(d));

        state.PauseTiming();
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));
        state.ResumeTiming();
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
    destroyDeviceArray<int>(values);
}

void
stdgpu_deque_clear(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);
    int* values = create_values(deque_size);

    stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));

    for (auto _ : state)
    {
        d.clear();

        state.PauseTiming();
        stdgpu::for_each_index(stdgpu::execution::device, deque_size, push_back_deque<int>(d, values));
        state.ResumeTiming();
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
    destroyDeviceArray<int>(values);
}

void
stdgpu_deque_valid(benchmark::State& state, const stdgpu::index_t deque_size)
{
    stdgpu::deque<int> d = stdgpu::deque<int>::createDeviceObject(deque_size);

    benchmark::DoNotOptimize(d.valid());

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(d.valid());
    }

    stdgpu::deque<int>::destroyDeviceObject(d);
}

#define STDGPU_REGISTER_BENCHMARK(function)                                                                            \
    BENCHMARK_CAPTURE(function, 1000, 1000)->Unit(benchmark::kMillisecond);                                            \
    BENCHMARK_CAPTURE(function, 100000, 100000)->Unit(benchmark::kMillisecond);                                        \
    BENCHMARK_CAPTURE(function, 10000000, 10000000)->Unit(benchmark::kMillisecond);

STDGPU_REGISTER_BENCHMARK(stdgpu_deque_push_back)
STDGPU_REGISTER_BENCHMARK(stdgpu_deque_push_front)
STDGPU_REGISTER_BENCHMARK(stdgpu_deque_pop_back)
STDGPU_REGISTER_BENCHMARK(stdgpu_deque_pop_front)

// clear is significantly faster than non-measured push_back
#if STDGPU_BACKEND != STDGPU_BACKEND_OPENMP
STDGPU_REGISTER_BENCHMARK(stdgpu_deque_clear)
#endif

STDGPU_REGISTER_BENCHMARK(stdgpu_deque_valid)
