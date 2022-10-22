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

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <stdgpu/cstddef.h>

namespace test_utils
{
/**
 * \brief Returns a seed chosen "truly" at random
 * \return The randomly chosen seed
 */
inline std::size_t
random_seed()
{
    try
    {
        std::random_device rd("/dev/urandom");

        // rd.entropy() != 0.0
        if (std::abs(rd.entropy()) >= std::numeric_limits<double>::min())
        {
            return rd();
        }

        throw std::runtime_error("Entropy is 0.0");
    }
    // For some reason, the following code fails to compile with NVCC+MSVC using the CUDA backend:
    // [[maybe_unused]] const std::exception& e
    // Thus, use the version below to fix unused parameter warnings
    catch (const std::exception&)
    {
    }

    return static_cast<std::size_t>(std::chrono::system_clock::now().time_since_epoch().count());
}

/**
 * \brief Returns a seed chosen "truly" at random for the this local thread
 * \return The randomly chosen thread-local seed
 */
inline std::size_t
random_thread_seed()
{
    return random_seed() + std::hash<std::thread::id>()(std::this_thread::get_id());
}

/**
 * \brief Runs the given function in N threads where N is the number of available concurrent threads the CPU can handle
 * \param[in] f A function
 * \param[in] args The arguments passed to f
 */
template <typename F, typename... Args>
inline void
for_each_concurrent_thread(F&& f, Args&&... args)
{
    const stdgpu::index_t concurrent_threads =
            std::max<stdgpu::index_t>(1, static_cast<stdgpu::index_t>(std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(concurrent_threads));

    for (stdgpu::index_t i = 0; i < concurrent_threads; ++i)
    {
        threads.emplace_back(std::forward<F>(f), std::forward<Args>(args)...);
    }

    for (stdgpu::index_t i = 0; i < concurrent_threads; ++i)
    {
        if (threads[static_cast<std::size_t>(i)].joinable())
        {
            threads[static_cast<std::size_t>(i)].join();
        }
    }
}
} // namespace test_utils

#endif // TEST_UTILS_H
