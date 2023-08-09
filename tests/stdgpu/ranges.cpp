/*
 *  Copyright 2020 Patrick Stotko
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

#include <numeric>

#include <stdgpu/algorithm.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>
#include <stdgpu/ranges.h>

class stdgpu_ranges : public ::testing::Test
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

template class device_range<int>;

template class host_range<int>;

template class transform_range<device_range<int>, identity>;

} // namespace stdgpu

TEST_F(stdgpu_ranges, device_range_default)
{
    const stdgpu::device_range<int> array_range;
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, nullptr);
    EXPECT_EQ(array_end, nullptr);
    EXPECT_EQ(array_range.size(), 0);
    EXPECT_TRUE(array_range.empty());
}

TEST_F(stdgpu_ranges, device_range_pointer_with_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    const stdgpu::device_range<int> array_range(array, size);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, device_range_pointer_automatic_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    const stdgpu::device_range<int> array_range(array);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, device_range_iterator_with_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);
    stdgpu::device_ptr<int> begin_iterator = stdgpu::make_device(array);

    const stdgpu::device_range<int> array_range(begin_iterator, size);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, device_range_iterator_pair)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);
    stdgpu::device_ptr<int> begin_iterator = stdgpu::make_device(array);
    stdgpu::device_ptr<int> end_iterator = stdgpu::make_device(array + size);

    const stdgpu::device_range<int> array_range(begin_iterator, end_iterator);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, device_range_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    const stdgpu::device_range<int> array_range(array, size);

    EXPECT_EQ(array_range.size(), size);

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, device_range_empty)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    const stdgpu::device_range<int> array_range(array, size);

    EXPECT_FALSE(array_range.empty());

    destroyDeviceArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_default)
{
    const stdgpu::host_range<int> array_range;
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, nullptr);
    EXPECT_EQ(array_end, nullptr);
    EXPECT_EQ(array_range.size(), 0);
    EXPECT_TRUE(array_range.empty());
}

TEST_F(stdgpu_ranges, host_range_pointer_with_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array, size);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_pointer_automatic_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_iterator_with_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);
    stdgpu::host_ptr<int> begin_iterator = stdgpu::make_host(array);

    const stdgpu::host_range<int> array_range(begin_iterator, size);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_iterator_pair)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);
    stdgpu::host_ptr<int> begin_iterator = stdgpu::make_host(array);
    stdgpu::host_ptr<int> end_iterator = stdgpu::make_host(array + size);

    const stdgpu::host_range<int> array_range(begin_iterator, end_iterator);
    int* array_begin = array_range.begin().get();
    int* array_end = array_range.end().get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end, array + size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array, size);

    EXPECT_EQ(array_range.size(), size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, host_range_empty)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array, size);

    EXPECT_FALSE(array_range.empty());

    destroyHostArray<int>(array);
}

template <typename T>
struct square
{
    STDGPU_HOST_DEVICE T
    operator()(const T x) const
    {
        return x * x;
    }
};

TEST_F(stdgpu_ranges, transform_range_default)
{
    const stdgpu::transform_range<stdgpu::host_range<int>, square<int>> square_range;
    int* array_begin = square_range.begin().base().get();
    int* array_end = square_range.end().base().get();

    EXPECT_EQ(array_begin, nullptr);
    EXPECT_EQ(array_end, nullptr);
    EXPECT_EQ(square_range.size(), 0);
    EXPECT_TRUE(square_range.empty());
}

TEST_F(stdgpu_ranges, transform_range_with_range)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);
    int* array_result = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array);
    const stdgpu::transform_range<stdgpu::host_range<int>, square<int>> square_range(array_range);

    // Setup array
    std::iota(array_range.begin(), array_range.end(), 0);

    // Execute transformation and write into array_result
    stdgpu::copy(stdgpu::execution::host, square_range.begin(), square_range.end(), stdgpu::host_begin(array_result));

    for (stdgpu::index_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(array[i], i);
        EXPECT_EQ(array_result[i], i * i);
    }

    destroyHostArray<int>(array);
    destroyHostArray<int>(array_result);
}

TEST_F(stdgpu_ranges, transform_range_with_range_and_function)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);
    int* array_result = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array);
    const stdgpu::transform_range<stdgpu::host_range<int>, square<int>> square_range(array_range, square<int>());

    // Setup array
    std::iota(array_range.begin(), array_range.end(), 0);

    // Execute transformation and write into array_result
    stdgpu::copy(stdgpu::execution::host, square_range.begin(), square_range.end(), stdgpu::host_begin(array_result));

    for (stdgpu::index_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(array[i], i);
        EXPECT_EQ(array_result[i], i * i);
    }

    destroyHostArray<int>(array);
    destroyHostArray<int>(array_result);
}

TEST_F(stdgpu_ranges, transform_range_size)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array);
    const stdgpu::transform_range<stdgpu::host_range<int>, square<int>> square_range(array_range, square<int>());

    EXPECT_EQ(square_range.size(), size);

    destroyHostArray<int>(array);
}

TEST_F(stdgpu_ranges, transform_range_empty)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    const stdgpu::host_range<int> array_range(array);
    const stdgpu::transform_range<stdgpu::host_range<int>, square<int>> square_range(array_range, square<int>());

    EXPECT_FALSE(square_range.empty());

    destroyHostArray<int>(array);
}
