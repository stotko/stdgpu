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

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>

namespace benchmark_utils
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
} // namespace benchmark_utils

#endif // BENCHMARK_UTILS_H
