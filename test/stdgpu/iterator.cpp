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

#include <vector>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_iterator : public ::testing::Test
{
    protected:
        // Called before each test
        void SetUp() override
        {

        }

        // Called after each test
        void TearDown() override
        {

        }

};


// Explicit template instantiations
namespace stdgpu
{

template
device_ptr<int>
make_device<int>(int*);

template
host_ptr<int>
make_host<int>(int*);

template
index64_t
size(int*);

template
host_ptr<int>
host_begin<int>(int*);

template
host_ptr<int>
host_end<int>(int*);

template
device_ptr<int>
device_begin<int>(int*);

template
device_ptr<int>
device_end<int>(int*);

template
host_ptr<const int>
host_begin<int>(const int*);

template
host_ptr<const int>
host_end<int>(const int*);

template
device_ptr<const int>
device_begin<int>(const int*);

template
device_ptr<const int>
device_end<int>(const int*);

template
host_ptr<const int>
host_cbegin<int>(const int*);

template
host_ptr<const int>
host_cend<int>(const int*);

template
device_ptr<const int>
device_cbegin<int>(const int*);

template
device_ptr<const int>
device_cend<int>(const int*);

/*
template
auto
host_begin<vector<int>>(vector<int>& host_container) -> decltype(host_container.host_begin());

template
auto
host_end<vector<int>>(vector<int>& host_container) -> decltype(host_container.host_end());

template
auto
device_begin<vector<int>>(vector<int>& device_container) -> decltype(device_container.device_begin());

template
auto
device_end<vector<int>>(vector<int>& device_container) -> decltype(device_container.device_end());

template
auto
host_begin<vector<int>>(const vector<int>& host_container) -> decltype(host_container.host_begin());

template
auto
host_end<vector<int>>(const vector<int>& host_container) -> decltype(host_container.host_end());

template
auto
device_begin<vector<int>>(const vector<int>& device_container) -> decltype(device_container.device_begin());

template
auto
device_end<vector<int>>(const vector<int>& device_container) -> decltype(device_container.device_end());

template
auto
host_cbegin<vector<int>>(const vector<int>& host_container) -> decltype(host_container.host_cbegin());

template
auto
host_cend<vector<int>>(const vector<int>& host_container) -> decltype(host_container.host_cend());

template
auto
device_cbegin<vector<int>>(const vector<int>& device_container) -> decltype(device_container.device_cbegin());

template
auto
device_cend<vector<int>>(const vector<int>& device_container) -> decltype(device_container.device_cend());

template
class back_insert_iterator<deque<int>>;

template
class front_insert_iterator<deque<int>>;

template
class insert_iterator<unordered_set<int>>;
*/

} // namespace stdgpu


TEST_F(stdgpu_iterator, size_device_void)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<void*>(array)), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, size_host_void)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<void*>(array)), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyHostArray<int>(array);
}


TEST_F(stdgpu_iterator, size_nullptr_void)
{
    int* array = nullptr;
    EXPECT_EQ(stdgpu::size(reinterpret_cast<void*>(array)), static_cast<stdgpu::index64_t>(0));
}


TEST_F(stdgpu_iterator, size_device)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    EXPECT_EQ(stdgpu::size(array), size);

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, size_host)
{
    const stdgpu::index64_t size = 42;
    int* array = createHostArray<int>(size);

    EXPECT_EQ(stdgpu::size(array), size);

    destroyHostArray<int>(array);
}


TEST_F(stdgpu_iterator, size_nullptr)
{
    int* array = nullptr;
    EXPECT_EQ(stdgpu::size(array), static_cast<stdgpu::index64_t>(0));
}


TEST_F(stdgpu_iterator, size_device_shifted)
{
    const stdgpu::index64_t size = 42;
    int* array = createDeviceArray<int>(size);

    EXPECT_EQ(stdgpu::size(array + 24), static_cast<stdgpu::index64_t>(0));

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, size_host_shifted)
{
    const stdgpu::index64_t size = 42;
    int* array_result = createHostArray<int>(size);

    EXPECT_EQ(stdgpu::size(array_result + 24), static_cast<stdgpu::index64_t>(0));

    destroyHostArray<int>(array_result);
}


TEST_F(stdgpu_iterator, size_device_wrong_alignment)
{
    int* array = createDeviceArray<int>(1);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<std::size_t*>(array)), static_cast<stdgpu::index64_t>(0));

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, size_host_wrong_alignment)
{
    int* array_result = createHostArray<int>(1);

    EXPECT_EQ(stdgpu::size(reinterpret_cast<std::size_t*>(array_result)), static_cast<stdgpu::index64_t>(0));

    destroyHostArray<int>(array_result);
}


TEST_F(stdgpu_iterator, device_begin_end)
{
    const stdgpu::index_t size = 42;
    int* array = createDeviceArray<int>(size);

    int* array_begin   = stdgpu::device_begin(array).get();
    int* array_end     = stdgpu::device_end(  array).get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end,   array + size);

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, host_begin_end)
{
    const stdgpu::index_t size = 42;
    int* array_result = createHostArray<int>(size);

    int* array_result_begin   = stdgpu::host_begin(array_result).get();
    int* array_result_end     = stdgpu::host_end(  array_result).get();

    EXPECT_EQ(array_result_begin, array_result);
    EXPECT_EQ(array_result_end,   array_result + size);

    destroyHostArray<int>(array_result);
}


TEST_F(stdgpu_iterator, device_begin_end_const)
{
    const stdgpu::index_t size = 42;
    int* array = createDeviceArray<int>(size);

    const int* array_begin   = stdgpu::device_begin(reinterpret_cast<const int*>(array)).get();
    const int* array_end     = stdgpu::device_end(  reinterpret_cast<const int*>(array)).get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end,   array + size);

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, host_begin_end_const)
{
    const stdgpu::index_t size = 42;
    int* array_result = createHostArray<int>(size);

    const int* array_result_begin   = stdgpu::host_begin(reinterpret_cast<const int*>(array_result)).get();
    const int* array_result_end     = stdgpu::host_end(  reinterpret_cast<const int*>(array_result)).get();

    EXPECT_EQ(array_result_begin, array_result);
    EXPECT_EQ(array_result_end,   array_result + size);

    destroyHostArray<int>(array_result);
}


TEST_F(stdgpu_iterator, device_cbegin_cend)
{
    const stdgpu::index_t size = 42;
    int* array = createDeviceArray<int>(size);

    const int* array_begin   = stdgpu::device_cbegin(array).get();
    const int* array_end     = stdgpu::device_cend(  array).get();

    EXPECT_EQ(array_begin, array);
    EXPECT_EQ(array_end,   array + size);

    destroyDeviceArray<int>(array);
}


TEST_F(stdgpu_iterator, host_cbegin_cend)
{
    const stdgpu::index_t size = 42;
    int* array_result = createHostArray<int>(size);

    const int* array_result_begin   = stdgpu::host_cbegin(array_result).get();
    const int* array_result_end     = stdgpu::host_cend(  array_result).get();

    EXPECT_EQ(array_result_begin, array_result);
    EXPECT_EQ(array_result_end,   array_result + size);

    destroyHostArray<int>(array_result);
}


class back_insert_interface
{
    public:
        using value_type = std::vector<int>::value_type;

        explicit back_insert_interface(std::vector<int>& vector)
            : _vector(vector)
        {

        }

        void
        push_back(const int x)
        {
            _vector.push_back(x);
        }

    private:
        std::vector<int>& _vector;
};


class front_insert_interface
{
    public:
        using value_type = std::vector<int>::value_type;

        explicit front_insert_interface(std::vector<int>& vector)
            : _vector(vector)
        {

        }

        void
        push_front(const int x)
        {
            _vector.push_back(x);
        }

    private:
        std::vector<int>& _vector;
};


class insert_interface
{
    public:
        using value_type = std::vector<int>::value_type;

        explicit insert_interface(std::vector<int>& vector)
            : _vector(vector)
        {

        }

        void
        insert(const int x)
        {
            _vector.push_back(x);
        }

    private:
        std::vector<int>& _vector;
};


TEST_F(stdgpu_iterator, back_inserter)
{
    const stdgpu::index_t N = 100000;

    int* array = createHostArray<int>(N);
    std::vector<int> numbers;

    thrust::sequence(stdgpu::host_begin(array), stdgpu::host_end(array),
                     1);

    back_insert_interface ci(numbers);
    thrust::copy(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                 stdgpu::back_inserter(ci));

    int* array_result = copyCreateHost2HostArray<int>(numbers.data(), N, MemoryCopy::NO_CHECK);

    thrust::sort(stdgpu::host_begin(array_result), stdgpu::host_end(array_result));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_result[i], i + 1);
    }

    destroyHostArray<int>(array_result);
    destroyHostArray<int>(array);
}


TEST_F(stdgpu_iterator, front_inserter)
{
    const stdgpu::index_t N = 100000;

    int* array = createHostArray<int>(N);
    std::vector<int> numbers;

    thrust::sequence(stdgpu::host_begin(array), stdgpu::host_end(array),
                     1);

    front_insert_interface ci(numbers);
    thrust::copy(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                 stdgpu::front_inserter(ci));

    int* array_result = copyCreateHost2HostArray<int>(numbers.data(), N, MemoryCopy::NO_CHECK);

    thrust::sort(stdgpu::host_begin(array_result), stdgpu::host_end(array_result));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_result[i], i + 1);
    }

    destroyHostArray<int>(array_result);
    destroyHostArray<int>(array);
}


TEST_F(stdgpu_iterator, inserter)
{
    const stdgpu::index_t N = 100000;

    int* array = createHostArray<int>(N);
    std::vector<int> numbers;

    thrust::sequence(stdgpu::host_begin(array), stdgpu::host_end(array),
                     1);

    insert_interface ci(numbers);
    thrust::copy(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                 stdgpu::inserter(ci));

    int* array_result = copyCreateHost2HostArray<int>(numbers.data(), N, MemoryCopy::NO_CHECK);

    thrust::sort(stdgpu::host_begin(array_result), stdgpu::host_end(array_result));

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(array_result[i], i + 1);
    }

    destroyHostArray<int>(array_result);
    destroyHostArray<int>(array);
}


