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
#include <stdgpu/memory.h>
#include <stdgpu/vector.cuh>

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
class push_back_vector
{
public:
    push_back_vector(const stdgpu::vector<T>& pool, T* values)
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
    T* _values;
};

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
} // namespace

void
stdgpu_vector_insert(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);
    int* values = create_values(vector_size);

    v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));
    v.clear();

    for (auto _ : state)
    {
        v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

        state.PauseTiming();
        v.clear();
        state.ResumeTiming();
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
    destroyDeviceArray<int>(values);
}

void
stdgpu_vector_erase(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);
    int* values = create_values(vector_size);

    v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

    for (auto _ : state)
    {
        v.erase(v.device_begin(), v.device_end());

        state.PauseTiming();
        v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));
        state.ResumeTiming();
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
    destroyDeviceArray<int>(values);
}

void
stdgpu_vector_push_back(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);
    int* values = create_values(vector_size);

    stdgpu::for_each_index(stdgpu::execution::device, vector_size, push_back_vector<int>(v, values));
    v.clear();

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, vector_size, push_back_vector<int>(v, values));

        state.PauseTiming();
        v.clear();
        state.ResumeTiming();
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
    destroyDeviceArray<int>(values);
}

void
stdgpu_vector_pop_back(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);
    int* values = create_values(vector_size);

    v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

    for (auto _ : state)
    {
        stdgpu::for_each_index(stdgpu::execution::device, vector_size, pop_back_vector<int>(v));

        state.PauseTiming();
        v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));
        state.ResumeTiming();
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
    destroyDeviceArray<int>(values);
}

void
stdgpu_vector_clear(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);
    int* values = create_values(vector_size);

    v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));

    for (auto _ : state)
    {
        v.clear();

        state.PauseTiming();
        v.insert(v.device_end(), stdgpu::device_begin(values), stdgpu::device_end(values));
        state.ResumeTiming();
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
    destroyDeviceArray<int>(values);
}

void
stdgpu_vector_valid(benchmark::State& state, const stdgpu::index_t vector_size)
{
    stdgpu::vector<int> v = stdgpu::vector<int>::createDeviceObject(vector_size);

    benchmark::DoNotOptimize(v.valid());

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(v.valid());
    }

    stdgpu::vector<int>::destroyDeviceObject(v);
}

#define STDGPU_REGISTER_BENCHMARK(function)                                                                            \
    BENCHMARK_CAPTURE(function, 1000, 1000)->Unit(benchmark::kMillisecond);                                            \
    BENCHMARK_CAPTURE(function, 100000, 100000)->Unit(benchmark::kMillisecond);                                        \
    BENCHMARK_CAPTURE(function, 10000000, 10000000)->Unit(benchmark::kMillisecond);

STDGPU_REGISTER_BENCHMARK(stdgpu_vector_insert)
STDGPU_REGISTER_BENCHMARK(stdgpu_vector_erase)
STDGPU_REGISTER_BENCHMARK(stdgpu_vector_push_back)
STDGPU_REGISTER_BENCHMARK(stdgpu_vector_pop_back)

// clear is significantly faster than non-measured insert
#if STDGPU_BACKEND != STDGPU_BACKEND_OPENMP
STDGPU_REGISTER_BENCHMARK(stdgpu_vector_clear)
#endif

STDGPU_REGISTER_BENCHMARK(stdgpu_vector_valid)
