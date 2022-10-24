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

#include <gtest/gtest.h>

#include <vector>

#include <stdgpu/functional.h>
#include <stdgpu/numeric.h>
#include <stdgpu/platform.h>

class stdgpu_numeric : public ::testing::Test
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

TEST_F(stdgpu_numeric, iota)
{
    const stdgpu::index_t N = static_cast<stdgpu::index_t>(pow(2, 22));
    std::vector<stdgpu::index_t> indices_vector(static_cast<std::size_t>(N));
    stdgpu::index_t* indices = indices_vector.data();

    const stdgpu::index_t init = 42;
    stdgpu::iota(stdgpu::execution::host, indices_vector.begin(), indices_vector.end(), init);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(indices[i], i + init);
    }
}

class number_sequence
{
public:
    explicit number_sequence(std::int64_t* numbers)
      : _numbers(numbers)
    {
    }

    STDGPU_HOST_DEVICE std::int64_t
    operator()(const stdgpu::index_t i) const
    {
        return _numbers[i];
    }

private:
    std::int64_t* _numbers;
};

TEST_F(stdgpu_numeric, transform_reduce_index)
{
    const stdgpu::index_t N = static_cast<stdgpu::index_t>(pow(2, 22));
    std::vector<std::int64_t> numbers_vector(static_cast<std::size_t>(N));
    std::int64_t* numbers = numbers_vector.data();

    const std::int64_t init = 42;
    stdgpu::iota(stdgpu::execution::host, numbers_vector.begin(), numbers_vector.end(), init);

    const std::int64_t shift = 21;
    std::int64_t shifted_sum = stdgpu::transform_reduce_index(stdgpu::execution::host,
                                                              N,
                                                              shift,
                                                              stdgpu::plus<>(),
                                                              number_sequence(numbers));

    auto sum_closed_form = [](const std::int64_t n) { return n * (n + 1) / 2; };

    EXPECT_EQ(shifted_sum,
              shift + sum_closed_form(init - 1 + static_cast<std::int64_t>(N)) - sum_closed_form(init - 1));
}
