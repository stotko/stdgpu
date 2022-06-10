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

#include <stdgpu/bitset.cuh>

void
stdgpu_bitset_set(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    bitset.reset();

    for (auto _ : state)
    {
        bitset.set();

        state.PauseTiming();
        bitset.reset();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_reset(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    bitset.set();

    for (auto _ : state)
    {
        bitset.reset();

        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_flip(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    bitset.set();

    for (auto _ : state)
    {
        bitset.flip();

        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_count(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    benchmark::DoNotOptimize(bitset.count());

    for (auto _ : state)
    {
        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();

        benchmark::DoNotOptimize(bitset.count());

        state.PauseTiming();
        bitset.reset();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_all(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    benchmark::DoNotOptimize(bitset.all());

    for (auto _ : state)
    {
        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();

        benchmark::DoNotOptimize(bitset.all());

        state.PauseTiming();
        bitset.reset();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_any(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    benchmark::DoNotOptimize(bitset.any());

    for (auto _ : state)
    {
        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();

        benchmark::DoNotOptimize(bitset.any());

        state.PauseTiming();
        bitset.reset();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

void
stdgpu_bitset_none(benchmark::State& state, const stdgpu::index_t bitset_size)
{
    stdgpu::bitset<> bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);

    benchmark::DoNotOptimize(bitset.none());

    for (auto _ : state)
    {
        state.PauseTiming();
        bitset.set();
        state.ResumeTiming();

        benchmark::DoNotOptimize(bitset.none());

        state.PauseTiming();
        bitset.reset();
        state.ResumeTiming();
    }

    stdgpu::bitset<>::destroyDeviceObject(bitset);
}

#define STDGPU_REGISTER_BENCHMARK(function)                                                                            \
    BENCHMARK_CAPTURE(function, 1000, 1000)->Unit(benchmark::kMillisecond);                                            \
    BENCHMARK_CAPTURE(function, 1000000, 1000000)->Unit(benchmark::kMillisecond);                                      \
    BENCHMARK_CAPTURE(function, 1000000000, 1000000000)->Unit(benchmark::kMillisecond);

STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_set)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_reset)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_flip)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_count)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_all)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_any)
STDGPU_REGISTER_BENCHMARK(stdgpu_bitset_none)
