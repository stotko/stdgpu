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

#include <stdgpu/numeric.h>

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
    const stdgpu::index_t N = 100000000;
    std::vector<stdgpu::index_t> indices_vector(N);
    stdgpu::index_t* indices = indices_vector.data();

    stdgpu::index_t init = 42;
    stdgpu::iota(thrust::host, indices_vector.begin(), indices_vector.end(), init);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(indices[i], i + init);
    }
}
