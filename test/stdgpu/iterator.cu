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

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/vector.cuh>



class stdgpu_iterator : public ::testing::Test
{
    protected:
        // Called before each test
        virtual void SetUp()
        {

        }

        // Called after each test
        virtual void TearDown()
        {

        }

};


TEST_F(stdgpu_iterator, size_device_void)
{
    int* array_device = createDeviceArray<int>(42);

    EXPECT_EQ(stdgpu::size((void*)array_device), 42 * sizeof(int));

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, size_host_void)
{
    int* array_host = createHostArray<int>(42);

    EXPECT_EQ(stdgpu::size((void*)array_host), 42 * sizeof(int));

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, size_nullptr_void)
{
    EXPECT_EQ(stdgpu::size((void*)nullptr), static_cast<size_t>(0));
}


TEST_F(stdgpu_iterator, size_device)
{
    int* array_device = createDeviceArray<int>(42);

    EXPECT_EQ(stdgpu::size(array_device), static_cast<size_t>(42));

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, size_host)
{
    int* array_host = createHostArray<int>(42);

    EXPECT_EQ(stdgpu::size(array_host), static_cast<size_t>(42));

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, size_nullptr)
{
    EXPECT_EQ(stdgpu::size((int*)nullptr), static_cast<size_t>(0));
}


TEST_F(stdgpu_iterator, size_device_shifted)
{
    int* array_device = createDeviceArray<int>(42);

    EXPECT_EQ(stdgpu::size(array_device + 24), static_cast<size_t>(0));

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, size_host_shifted)
{
    int* array_host = createHostArray<int>(42);

    EXPECT_EQ(stdgpu::size(array_host + 24), static_cast<size_t>(0));

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, size_device_wrong_alignment)
{
    int* array_device = createDeviceArray<int>(1);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<size_t*>(array_device)), static_cast<size_t>(0));

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, size_host_wrong_alignment)
{
    int* array_host = createHostArray<int>(1);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<size_t*>(array_host)), static_cast<size_t>(0));

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, device_begin_end)
{
    int* array_device = createDeviceArray<int>(42);

    int* array_device_begin   = stdgpu::device_begin(array_device).get();
    int* array_device_end     = stdgpu::device_end(  array_device).get();

    EXPECT_EQ(array_device_begin, array_device);
    EXPECT_EQ(array_device_end,   array_device + 42);

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, host_begin_end)
{
    int* array_host = createHostArray<int>(42);

    int* array_host_begin   = stdgpu::host_begin(array_host).get();
    int* array_host_end     = stdgpu::host_end(  array_host).get();

    EXPECT_EQ(array_host_begin, array_host);
    EXPECT_EQ(array_host_end,   array_host + 42);

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, device_begin_end_const)
{
    int* array_device = createDeviceArray<int>(42);

    const int* array_device_begin   = stdgpu::device_begin(reinterpret_cast<const int*>(array_device)).get();
    const int* array_device_end     = stdgpu::device_end(  reinterpret_cast<const int*>(array_device)).get();

    EXPECT_EQ(array_device_begin, array_device);
    EXPECT_EQ(array_device_end,   array_device + 42);

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, host_begin_end_const)
{
    int* array_host = createHostArray<int>(42);

    const int* array_host_begin   = stdgpu::host_begin(reinterpret_cast<const int*>(array_host)).get();
    const int* array_host_end     = stdgpu::host_end(  reinterpret_cast<const int*>(array_host)).get();

    EXPECT_EQ(array_host_begin, array_host);
    EXPECT_EQ(array_host_end,   array_host + 42);

    destroyHostArray<int>(array_host);
}


TEST_F(stdgpu_iterator, device_cbegin_cend)
{
    int* array_device = createDeviceArray<int>(42);

    const int* array_device_begin   = stdgpu::device_cbegin(array_device).get();
    const int* array_device_end     = stdgpu::device_cend(  array_device).get();

    EXPECT_EQ(array_device_begin, array_device);
    EXPECT_EQ(array_device_end,   array_device + 42);

    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, host_cbegin_cend)
{
    int* array_host = createHostArray<int>(42);

    const int* array_host_begin   = stdgpu::host_cbegin(array_host).get();
    const int* array_host_end     = stdgpu::host_cend(  array_host).get();

    EXPECT_EQ(array_host_begin, array_host);
    EXPECT_EQ(array_host_end,   array_host + 42);

    destroyHostArray<int>(array_host);
}


struct back_insert_interface
{
    using value_type = stdgpu::vector<int>::value_type;

    back_insert_interface(const stdgpu::vector<int>& vector)
        : vector(vector)
    {

    }

    inline __device__ void
    push_back(const int x)
    {
        vector.push_back(x);
    }

    stdgpu::vector<int> vector;
};


struct front_insert_interface
{
    using value_type = stdgpu::vector<int>::value_type;

    front_insert_interface(const stdgpu::vector<int>& vector)
        : vector(vector)
    {

    }

    inline __device__ void
    push_front(const int x)
    {
        vector.push_back(x);
    }

    stdgpu::vector<int> vector;
};


struct insert_interface
{
    using value_type = stdgpu::vector<int>::value_type;

    insert_interface(const stdgpu::vector<int>& vector)
        : vector(vector)
    {

    }

    inline __device__ void
    insert(const int x)
    {
        vector.push_back(x);
    }

    stdgpu::vector<int> vector;
};


TEST_F(stdgpu_iterator, back_inserter)
{
    const stdgpu::index_t N = 100000;

    int* array_device = createDeviceArray<int>(N);
    stdgpu::vector<int> numbers = stdgpu::vector<int>::createDeviceObject(N);

    thrust::sequence(stdgpu::device_begin(array_device), stdgpu::device_end(array_device),
                     1);

    back_insert_interface ci(numbers);
    thrust::copy(stdgpu::device_cbegin(array_device), stdgpu::device_cend(array_device),
                 stdgpu::back_inserter(ci));

    int* array_host = copyCreateDevice2HostArray<int>(numbers.data(), N);

    thrust::sort(stdgpu::host_begin(array_host), stdgpu::host_end(array_host));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_host[i], i + 1);
    }

    destroyHostArray<int>(array_host);
    stdgpu::vector<int>::destroyDeviceObject(numbers);
    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, front_inserter)
{
    const stdgpu::index_t N = 100000;

    int* array_device = createDeviceArray<int>(N);
    stdgpu::vector<int> numbers = stdgpu::vector<int>::createDeviceObject(N);

    thrust::sequence(stdgpu::device_begin(array_device), stdgpu::device_end(array_device),
                     1);

    front_insert_interface ci(numbers);
    thrust::copy(stdgpu::device_cbegin(array_device), stdgpu::device_cend(array_device),
                 stdgpu::front_inserter(ci));

    int* array_host = copyCreateDevice2HostArray<int>(numbers.data(), N);

    thrust::sort(stdgpu::host_begin(array_host), stdgpu::host_end(array_host));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_host[i], i + 1);
    }

    destroyHostArray<int>(array_host);
    stdgpu::vector<int>::destroyDeviceObject(numbers);
    destroyDeviceArray<int>(array_device);
}


TEST_F(stdgpu_iterator, inserter)
{
    const stdgpu::index_t N = 100000;

    int* array_device = createDeviceArray<int>(N);
    stdgpu::vector<int> numbers = stdgpu::vector<int>::createDeviceObject(N);

    thrust::sequence(stdgpu::device_begin(array_device), stdgpu::device_end(array_device),
                     1);

    insert_interface ci(numbers);
    thrust::copy(stdgpu::device_cbegin(array_device), stdgpu::device_cend(array_device),
                 stdgpu::inserter(ci));

    int* array_host = copyCreateDevice2HostArray<int>(numbers.data(), N);

    thrust::sort(stdgpu::host_begin(array_host), stdgpu::host_end(array_host));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_host[i], i + 1);
    }

    destroyHostArray<int>(array_host);
    stdgpu::vector<int>::destroyDeviceObject(numbers);
    destroyDeviceArray<int>(array_device);
}


