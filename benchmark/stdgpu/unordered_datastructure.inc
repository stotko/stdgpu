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

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS
    #error "Class name for benchmark not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_TYPE
    #error "Data structure type not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE
    #error "Key to Value conversion not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY
    #error "Value to Key conversion not specified!"
#endif

#include <benchmark/benchmark.h>

#include <algorithm>
#include <limits>
#include <random>

#include <benchmark_utils.h>
#include <stdgpu/algorithm.h>
#include <stdgpu/impl/preprocessor.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>

// convenience wrapper to improve readability
using benchmark_unordered_datastructure = STDGPU_UNORDERED_DATASTRUCTURE_TYPE;

namespace
{
class Key2ValueFunctor
{
public:
    Key2ValueFunctor(benchmark_unordered_datastructure::key_type* keys,
                     benchmark_unordered_datastructure::value_type* values)
      : _keys(keys)
      , _values(values)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const stdgpu::index_t i)
    {
        stdgpu::construct_at(&(_values[i]), STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_keys[i]));
    }

private:
    benchmark_unordered_datastructure::key_type* _keys;
    benchmark_unordered_datastructure::value_type* _values;
};

benchmark_unordered_datastructure::value_type*
create_values(const stdgpu::index_t N)
{
    // Generate true random numbers
    size_t seed = benchmark_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<std::int16_t> dist(std::numeric_limits<std::int16_t>::lowest(),
                                                     std::numeric_limits<std::int16_t>::max());

    benchmark_unordered_datastructure::key_type* host_keys =
            createHostArray<benchmark_unordered_datastructure::key_type>(N);

    std::generate(host_keys, host_keys + N, [&dist, &rng]() {
        return benchmark_unordered_datastructure::key_type(dist(rng), dist(rng), dist(rng));
    });

    benchmark_unordered_datastructure::key_type* keys =
            copyCreateHost2DeviceArray<benchmark_unordered_datastructure::key_type>(host_keys, N);
    destroyHostArray<benchmark_unordered_datastructure::key_type>(host_keys);

    benchmark_unordered_datastructure::value_type* values =
            createDeviceArray<benchmark_unordered_datastructure::value_type>(N);

    stdgpu::for_each_index(stdgpu::execution::device, N, Key2ValueFunctor(keys, values));

    destroyDeviceArray<benchmark_unordered_datastructure::key_type>(keys);

    return values;
}

class Value2KeyFunctor
{
public:
    Value2KeyFunctor(benchmark_unordered_datastructure::key_type* keys,
                     benchmark_unordered_datastructure::value_type* values)
      : _keys(keys)
      , _values(values)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const stdgpu::index_t i) const
    {
        _keys[i] = STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY(_values[i]);
    }

private:
    benchmark_unordered_datastructure::key_type* _keys;
    benchmark_unordered_datastructure::value_type* _values;
};

benchmark_unordered_datastructure::key_type*
extract_keys(benchmark_unordered_datastructure::value_type* values, const stdgpu::index_t N)
{
    benchmark_unordered_datastructure::key_type* keys =
            createDeviceArray<benchmark_unordered_datastructure::key_type>(N);

    stdgpu::for_each_index(stdgpu::execution::device, N, Value2KeyFunctor(keys, values));

    return keys;
}
} // namespace

void STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _insert)(benchmark::State& state,
                                                                                 const stdgpu::index_t unordered_size)
{
    // Use larger container to avoid unlikely but possible overflow of excess list
    benchmark_unordered_datastructure u = benchmark_unordered_datastructure::createDeviceObject(2 * unordered_size);
    benchmark_unordered_datastructure::value_type* values = create_values(unordered_size);

    u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));
    u.clear();

    for (auto _ : state)
    {
        u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));

        state.PauseTiming();
        u.clear();
        state.ResumeTiming();
    }

    benchmark_unordered_datastructure::destroyDeviceObject(u);
    destroyDeviceArray<benchmark_unordered_datastructure::value_type>(values);
}

void STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _erase)(benchmark::State& state,
                                                                                const stdgpu::index_t unordered_size)
{
    // Use larger container to avoid unlikely but possible overflow of excess list
    benchmark_unordered_datastructure u = benchmark_unordered_datastructure::createDeviceObject(2 * unordered_size);
    benchmark_unordered_datastructure::value_type* values = create_values(unordered_size);
    benchmark_unordered_datastructure::key_type* keys = extract_keys(values, unordered_size);

    u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));

    for (auto _ : state)
    {
        u.erase(stdgpu::device_begin(keys), stdgpu::device_end(keys));

        state.PauseTiming();
        u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));
        state.ResumeTiming();
    }

    benchmark_unordered_datastructure::destroyDeviceObject(u);
    destroyDeviceArray<benchmark_unordered_datastructure::value_type>(values);
    destroyDeviceArray<benchmark_unordered_datastructure::key_type>(keys);
}

void STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _clear)(benchmark::State& state,
                                                                                const stdgpu::index_t unordered_size)
{
    // Use larger container to avoid unlikely but possible overflow of excess list
    benchmark_unordered_datastructure u = benchmark_unordered_datastructure::createDeviceObject(2 * unordered_size);
    benchmark_unordered_datastructure::value_type* values = create_values(unordered_size);

    u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));

    for (auto _ : state)
    {
        u.clear();

        state.PauseTiming();
        u.insert(stdgpu::device_begin(values), stdgpu::device_end(values));
        state.ResumeTiming();
    }

    benchmark_unordered_datastructure::destroyDeviceObject(u);
    destroyDeviceArray<benchmark_unordered_datastructure::value_type>(values);
}

void STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _valid)(benchmark::State& state,
                                                                                const stdgpu::index_t vector_size)
{
    benchmark_unordered_datastructure u = benchmark_unordered_datastructure::createDeviceObject(vector_size);

    benchmark::DoNotOptimize(u.valid());

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(u.valid());
    }

    benchmark_unordered_datastructure::destroyDeviceObject(u);
}

#define STDGPU_REGISTER_BENCHMARK(function)                                                                            \
    BENCHMARK_CAPTURE(function, 1000, 1000)->Unit(benchmark::kMillisecond);                                            \
    BENCHMARK_CAPTURE(function, 100000, 100000)->Unit(benchmark::kMillisecond);                                        \
    BENCHMARK_CAPTURE(function, 10000000, 10000000)->Unit(benchmark::kMillisecond);

STDGPU_REGISTER_BENCHMARK(STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _insert))
STDGPU_REGISTER_BENCHMARK(STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _erase))

// clear is significantly faster than non-measured insert
#if STDGPU_BACKEND != STDGPU_BACKEND_OPENMP
STDGPU_REGISTER_BENCHMARK(STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _clear))
#endif

STDGPU_REGISTER_BENCHMARK(STDGPU_DETAIL_CAT2(STDGPU_UNORDERED_DATASTRUCTURE_BENCHMARK_CLASS, _valid))
