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

#ifndef STDGPU_MEMORY_TEST_CLASS
    #error "Class name for unit test not specified!"
#endif

#include <gtest/gtest.h>

#include <cmath>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <test_utils.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>


class STDGPU_MEMORY_TEST_CLASS : public ::testing::Test
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


class equal_to_number
{
    public:
        explicit equal_to_number(const int number)
            : _number(number)
        {

        }

        STDGPU_HOST_DEVICE bool
        operator()(const int value)
        {
            return (value == _number);
        }

    private:
        int _number;
};


// Explicit template instantiations
template
int*
createDeviceArray<int>(const stdgpu::index64_t,
                       const int);

template
int*
createHostArray<int>(const stdgpu::index64_t,
                     const int);

template
int*
createManagedArray<int>(const stdgpu::index64_t,
                        const int,
                        const Initialization);

template
void
destroyDeviceArray<int>(int*&);

template
void
destroyHostArray<int>(int*&);

template
void
destroyManagedArray<int>(int*&);

template
int*
copyCreateDevice2HostArray<int>(const int*,
                                const stdgpu::index64_t,
                                const MemoryCopy);

template
int*
copyCreateHost2DeviceArray<int>(const int*,
                                const stdgpu::index64_t,
                                const MemoryCopy);

template
int*
copyCreateHost2HostArray<int>(const int*,
                              const stdgpu::index64_t,
                              const MemoryCopy);

template
int*
copyCreateDevice2DeviceArray<int>(const int*,
                                  const stdgpu::index64_t,
                                  const MemoryCopy);

template
void
copyDevice2HostArray<int>(const int*,
                          const stdgpu::index64_t,
                          int*,
                          const MemoryCopy);

template
void
copyHost2DeviceArray<int>(const int*,
                          const stdgpu::index64_t,
                          int*,
                          const MemoryCopy);

template
void
copyHost2HostArray<int>(const int*,
                        const stdgpu::index64_t,
                        int*,
                        const MemoryCopy);

template
void
copyDevice2DeviceArray<int>(const int*,
                            const stdgpu::index64_t,
                            int*,
                            const MemoryCopy);

namespace stdgpu
{

template
struct safe_device_allocator<int>;

template
struct safe_host_allocator<int>;

template
struct safe_managed_allocator<int>;

template
struct allocator_traits<safe_device_allocator<int>>;

template
STDGPU_HOST_DEVICE void
allocator_traits<safe_device_allocator<int>>::construct<int>(safe_device_allocator<int>&,
                                                             int*);

template
STDGPU_HOST_DEVICE void
allocator_traits<safe_device_allocator<int>>::destroy<int>(safe_device_allocator<int>&,
                                                           int*);

template
STDGPU_HOST_DEVICE int*
construct_at<int>(int*);

template
STDGPU_HOST_DEVICE void
destroy_at<int>(int*);

template
void
destroy<int*>(int*,
              int*);

template
int*
destroy_n<int*, index_t>(int*,
                         index_t);

template
dynamic_memory_type
get_dynamic_memory_type<int>(int*);

template
index64_t
size_bytes<int>(int*);

} // namespace stdgpu



TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_device)
{
    const stdgpu::index64_t size = 42;
    int* array_device = createDeviceArray<int>(size);

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(array_device), stdgpu::dynamic_memory_type::device);

    destroyDeviceArray<int>(array_device);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_host)
{
    const stdgpu::index64_t size = 42;
    int* array_host = createHostArray<int>(size);

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(array_host), stdgpu::dynamic_memory_type::host);

    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_managed_on_device)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::DEVICE);

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(array_managed), stdgpu::dynamic_memory_type::managed);

    destroyManagedArray<int>(array_managed);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_managed_on_host)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::HOST);

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(array_managed), stdgpu::dynamic_memory_type::managed);

    destroyManagedArray<int>(array_managed);
}



TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_invalid_pointer)
{
    int* array_invalid = reinterpret_cast<int*>(42); // NOLINT(readability-magic-numbers)

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(array_invalid), stdgpu::dynamic_memory_type::invalid);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_variable_pointer)
{
    int non_dynamic_array = 42; // NOLINT(readability-magic-numbers)

    EXPECT_EQ(stdgpu::get_dynamic_memory_type(&non_dynamic_array), stdgpu::dynamic_memory_type::invalid);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, dynamic_memory_type_nullptr)
{
    EXPECT_EQ(stdgpu::get_dynamic_memory_type<int*>(nullptr), stdgpu::dynamic_memory_type::invalid);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_device)
{
    const stdgpu::index64_t size = 42;
    int* array_device = createDeviceArray<int>(size);

    EXPECT_EQ(stdgpu::size_bytes(array_device), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyDeviceArray<int>(array_device);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_host)
{
    const stdgpu::index64_t size = 42;
    int* array_host = createHostArray<int>(size);

    EXPECT_EQ(stdgpu::size_bytes(array_host), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_managed_device)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::DEVICE);

    EXPECT_EQ(stdgpu::size_bytes(array_managed), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyManagedArray<int>(array_managed);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_manged_host)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::HOST);

    EXPECT_EQ(stdgpu::size_bytes(array_managed), size * static_cast<stdgpu::index64_t>(sizeof(int)));

    destroyManagedArray<int>(array_managed);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_nullptr)
{
    EXPECT_EQ(stdgpu::size_bytes<int*>(nullptr), static_cast<stdgpu::index64_t>(0));
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_device_shifted)
{
    const stdgpu::index64_t size = 42;
    int* array_device = createDeviceArray<int>(size);

    const stdgpu::index64_t offset = 24;
    EXPECT_EQ(stdgpu::size_bytes(array_device + offset), static_cast<stdgpu::index64_t>(0));

    destroyDeviceArray<int>(array_device);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_host_shifted)
{
    const stdgpu::index64_t size = 42;
    int* array_host = createHostArray<int>(size);

    const stdgpu::index64_t offset = 24;
    EXPECT_EQ(stdgpu::size_bytes(array_host + offset), static_cast<stdgpu::index64_t>(0));

    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_managed_device_shifted)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::DEVICE);

    const stdgpu::index64_t offset = 24;
    EXPECT_EQ(stdgpu::size_bytes(array_managed + offset), static_cast<stdgpu::index64_t>(0));

    destroyManagedArray<int>(array_managed);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, size_bytes_managed_host_shifted)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 0;
    int* array_managed = createManagedArray<int>(size, default_value, Initialization::HOST);

    const stdgpu::index64_t offset = 24;
    EXPECT_EQ(stdgpu::size_bytes(array_managed + offset), static_cast<stdgpu::index64_t>(0));

    destroyManagedArray<int>(array_managed);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyDeviceArray_empty)
{
    int* array_device = createDeviceArray<int>(0, 0);

    EXPECT_EQ(array_device, nullptr);

    destroyDeviceArray<int>(array_device);

    EXPECT_EQ(array_device, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyHostArray_empty)
{
    int* array_host = createHostArray<int>(0, 0);

    EXPECT_EQ(array_host, nullptr);

    destroyHostArray<int>(array_host);

    EXPECT_EQ(array_host, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyManagedArray_empty)
{
    int* array_managed_device = createManagedArray<int>(0, 0, Initialization::DEVICE);
    int* array_managed_host = createManagedArray<int>(0, 0, Initialization::HOST);

    EXPECT_EQ(array_managed_device, nullptr);
    EXPECT_EQ(array_managed_host, nullptr);

    destroyManagedArray<int>(array_managed_device);
    destroyManagedArray<int>(array_managed_host);

    EXPECT_EQ(array_managed_device, nullptr);
    EXPECT_EQ(array_managed_host, nullptr);
}


namespace
{
    void createAndDestroyDeviceFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array_device = createDeviceArray<int>(size, default_value);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::all_of(stdgpu::device_cbegin(array_device), stdgpu::device_cend(array_device),
                                            equal_to_number(default_value)) );
            #endif

            destroyDeviceArray<int>(array_device);

            EXPECT_EQ(array_device, nullptr);
        }
    }

    void createAndDestroyHostFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array_host = createHostArray<int>(size, default_value);

            EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                        equal_to_number(default_value)) );


            destroyHostArray<int>(array_host);

            EXPECT_EQ(array_host, nullptr);
        }
    }

    void createAndDestroyManagedFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array_managed_device = createManagedArray<int>(size, default_value, Initialization::DEVICE);
            int* array_managed_host = createManagedArray<int>(size, default_value, Initialization::HOST);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::all_of(stdgpu::device_cbegin(array_managed_device), stdgpu::device_cend(array_managed_device),
                                            equal_to_number(default_value)) );
            #endif

            EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array_managed_host), stdgpu::host_cend(array_managed_host),
                                        equal_to_number(default_value)) );

            destroyManagedArray<int>(array_managed_device);
            destroyManagedArray<int>(array_managed_host);

            EXPECT_EQ(array_managed_device, nullptr);
            EXPECT_EQ(array_managed_host, nullptr);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyDeviceArray)
{
    createAndDestroyDeviceFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyDeviceArray_const_type)
{
    using T = thrust::pair<const int,float>;
    const T default_value = {10, 2.0F};
    const stdgpu::index64_t size = 42;

    T* array_device = createDeviceArray<T>(size, default_value);

    destroyDeviceArray<T>(array_device);

    EXPECT_EQ(array_device, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyDeviceArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&createAndDestroyDeviceFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyHostArray)
{
    createAndDestroyHostFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyHostArray_const_type)
{
    using T = thrust::pair<const int,float>;
    const T default_value = {10, 2.0F};
    const stdgpu::index64_t size = 42;

    T* array_host = createHostArray<T>(size, default_value);

    destroyHostArray<T>(array_host);

    EXPECT_EQ(array_host, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyHostArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&createAndDestroyHostFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyManagedArray)
{
    createAndDestroyManagedFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyManagedArray_const_type)
{
    using T = thrust::pair<const int,float>;
    const T default_value = {10, 2.0F};
    const stdgpu::index64_t size = 42;

    T* array_managed = createManagedArray<T>(size, default_value);

    destroyManagedArray<T>(array_managed);

    EXPECT_EQ(array_managed, nullptr);
}


/*
TEST_F(STDGPU_MEMORY_TEST_CLASS, createDestroyManagedArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&createAndDestroyManagedFunction,
                                           iterations_per_thread);
}
*/


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2HostArray_empty)
{
    int* array_host = createHostArray<int>(0, 0);

    EXPECT_EQ(array_host, nullptr);

    int* array_host_copy = copyCreateHost2HostArray<int>(array_host, 0);

    EXPECT_EQ(array_host_copy, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2HostArray_empty)
{
    int* array_device = createDeviceArray<int>(0, 0);

    EXPECT_EQ(array_device, nullptr);

    int* array_host_copy = copyCreateDevice2HostArray<int>(array_device, 0);

    EXPECT_EQ(array_host_copy, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2DeviceArray_empty)
{
    int* array_host = createHostArray<int>(  0, 0);

    EXPECT_EQ(array_host, nullptr);

    int* array_device_copy = copyCreateHost2DeviceArray<int>(array_host, 0);

    EXPECT_EQ(array_device_copy, nullptr);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2DeviceArray_empty)
{
    int* array_device = createDeviceArray<int>(0, 0);

    EXPECT_EQ(array_device, nullptr);

    int* array_device_copy = copyCreateDevice2DeviceArray<int>(array_device, 0);

    EXPECT_EQ(array_device_copy, nullptr);
}


namespace
{
    void copyCreateDevice2DeviceFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_copy = copyCreateDevice2DeviceArray<int>(array, size);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::equal(stdgpu::device_begin(array), stdgpu::device_end(array),
                                           stdgpu::device_begin(array_copy),
                                           thrust::equal_to<int>()) );
            #endif

            destroyDeviceArray<int>(array);
            destroyDeviceArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2DeviceArray)
{
    copyCreateDevice2DeviceFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2DeviceArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyCreateDevice2DeviceFunction,
                                           iterations_per_thread);
}


namespace
{
    void copyCreateHost2DeviceFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = copyCreateHost2DeviceArray<int>(array_host, size);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                           stdgpu::device_cbegin(array_copy),
                                           thrust::equal_to<int>()) );
            #endif

            destroyDeviceArray<int>(array);
            destroyHostArray<int>(array_host);
            destroyDeviceArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2DeviceArray)
{
    copyCreateHost2DeviceFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2DeviceArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyCreateHost2DeviceFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2DeviceArray_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array = createDeviceArray<int>(size, default_value);
    int* array_host = new int[static_cast<std::size_t>(size)];
    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        array_host[i] = default_value;
    }
    int* array_copy = copyCreateHost2DeviceArray<int>(array_host, size, MemoryCopy::NO_CHECK);

    #if STDGPU_DETAIL_IS_DEVICE_COMPILED
        EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                   stdgpu::device_cbegin(array_copy),
                                   thrust::equal_to<int>()) );
    #endif

    destroyDeviceArray<int>(array);
    delete[] array_host;
    destroyDeviceArray<int>(array_copy);
}


namespace
{
    void copyCreateDevice2HostFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = copyCreateDevice2HostArray<int>(array, size);

            EXPECT_TRUE( thrust::equal(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                       stdgpu::host_cbegin(array_copy),
                                       thrust::equal_to<int>()) );

            destroyDeviceArray<int>(array);
            destroyHostArray<int>(array_host);
            destroyHostArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2HostArray)
{
    copyCreateDevice2HostFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateDevice2HostArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyCreateDevice2HostFunction,
                                           iterations_per_thread);
}


namespace
{
    void copyCreateHost2HostFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = copyCreateHost2HostArray<int>(array_host, size);

            EXPECT_TRUE( thrust::equal(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                       stdgpu::host_cbegin(array_copy),
                                       thrust::equal_to<int>()) );

            destroyHostArray<int>(array_host);
            destroyHostArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2HostArray)
{
    copyCreateHost2HostFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2HostArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyCreateHost2HostFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyCreateHost2HostArray_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = new int[static_cast<std::size_t>(size)];
    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        array_host[i] = default_value;
    }
    int* array_copy = copyCreateHost2HostArray<int>(array_host, size, MemoryCopy::NO_CHECK);

    EXPECT_TRUE( thrust::equal(thrust::host,
                               array_host, array_host + size,
                               stdgpu::host_cbegin(array_copy),
                               thrust::equal_to<int>()) );

    delete[] array_host;
    destroyHostArray<int>(array_copy);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray_empty)
{
    int* array_host = createHostArray<int>(1, 0);

    int* array_host_copy = reinterpret_cast<int*>(42); // NOLINT(readability-magic-numbers)

    copyHost2HostArray<int>(array_host, 0, array_host_copy);
    EXPECT_EQ(array_host_copy, reinterpret_cast<int*>(42));

    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2HostArray_empty)
{
    int* array_device = createDeviceArray<int>(1, 0);

    int* array_host_copy = reinterpret_cast<int*>(42); // NOLINT(readability-magic-numbers)

    copyDevice2HostArray<int>(array_device, 0, array_host_copy);
    EXPECT_EQ(array_host_copy, reinterpret_cast<int*>(42));

    destroyDeviceArray<int>(array_device);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2DeviceArray_empty)
{
    int* array_host = createHostArray<int>(1, 0);

    int* array_device_copy = reinterpret_cast<int*>(42); // NOLINT(readability-magic-numbers)

    copyHost2DeviceArray<int>(array_host, 0, array_device_copy);
    EXPECT_EQ(array_device_copy, reinterpret_cast<int*>(42));
    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2DeviceArray_empty)
{
    int* array_device = createDeviceArray<int>(1, 0);

    int* array_device_copy = reinterpret_cast<int*>(42); // NOLINT(readability-magic-numbers)

    copyDevice2DeviceArray<int>(array_device, 0, array_device_copy);
    EXPECT_EQ(array_device_copy, reinterpret_cast<int*>(42));

    destroyDeviceArray<int>(array_device);
}


namespace
{
    void copyDevice2DeviceFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_copy = createDeviceArray<int>(size, 0);
            copyDevice2DeviceArray<int>(array, size, array_copy);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                           stdgpu::device_cbegin(array_copy),
                                           thrust::equal_to<int>()) );
            #endif

            destroyDeviceArray<int>(array);
            destroyDeviceArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2DeviceArray)
{
    copyDevice2DeviceFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2DeviceArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyDevice2DeviceFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2DeviceArray_self)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array = createDeviceArray<int>(size, default_value);
    int* array_copy = array;
    copyDevice2DeviceArray<int>(array, size, array_copy);

    #if STDGPU_DETAIL_IS_DEVICE_COMPILED
        EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                   stdgpu::device_cbegin(array_copy),
                                   thrust::equal_to<int>()) );
    #endif

    destroyDeviceArray<int>(array);
}


namespace
{
    void copyHost2DeviceFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = createDeviceArray<int>(size, 0);
            copyHost2DeviceArray<int>(array_host, size, array_copy);

            #if STDGPU_DETAIL_IS_DEVICE_COMPILED
                EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                           stdgpu::device_cbegin(array_copy),
                                           thrust::equal_to<int>()) );
            #endif

            destroyDeviceArray<int>(array);
            destroyHostArray<int>(array_host);
            destroyDeviceArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2DeviceArray)
{
    copyHost2DeviceFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2DeviceArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyHost2DeviceFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2DeviceArray_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array = createDeviceArray<int>(size, default_value);
    int* array_host = new int[static_cast<std::size_t>(size)];
    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        array_host[i] = default_value;
    }
    int* array_copy = createDeviceArray<int>(size, 0);
    copyHost2DeviceArray<int>(array_host, size, array_copy, MemoryCopy::NO_CHECK);

    #if STDGPU_DETAIL_IS_DEVICE_COMPILED
        EXPECT_TRUE( thrust::equal(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                   stdgpu::device_cbegin(array_copy),
                                   thrust::equal_to<int>()) );
    #endif

    destroyDeviceArray<int>(array);
    delete[] array_host;
    destroyDeviceArray<int>(array_copy);
}


namespace
{
    void copyDevice2HostFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array = createDeviceArray<int>(size, default_value);
            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = createHostArray<int>(size, 0);
            copyDevice2HostArray<int>(array, size, array_copy);

            EXPECT_TRUE( thrust::equal(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                       stdgpu::host_cbegin(array_copy),
                                       thrust::equal_to<int>()) );

            destroyDeviceArray<int>(array);
            destroyHostArray<int>(array_host);
            destroyHostArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2HostArray)
{
    copyDevice2HostFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2HostArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyDevice2HostFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyDevice2HostArray_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array = createDeviceArray<int>(size, default_value);
    int* array_host = createHostArray<int>(size, default_value);
    int* array_copy = new int[static_cast<std::size_t>(size)];
    copyDevice2HostArray<int>(array, size, array_copy, MemoryCopy::NO_CHECK);

    EXPECT_TRUE( thrust::equal(thrust::host,
                               stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                               array_copy,
                               thrust::equal_to<int>()) );

    destroyDeviceArray<int>(array);
    destroyHostArray<int>(array_host);
    delete[] array_copy;
}


namespace
{
    void copyHost2HostFunction(const stdgpu::index_t iterations)
    {
        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            const stdgpu::index64_t size = 42;
            const int default_value = 10;

            int* array_host = createHostArray<int>(size, default_value);
            int* array_copy = createHostArray<int>(size, 0);
            copyHost2HostArray<int>(array_host, size, array_copy);

            EXPECT_TRUE( thrust::equal(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                       stdgpu::host_cbegin(array_copy),
                                       thrust::equal_to<int>()) );

            destroyHostArray<int>(array_host);
            destroyHostArray<int>(array_copy);
        }
    }
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray)
{
    copyHost2HostFunction(1);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray_parallel)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 7));

    test_utils::for_each_concurrent_thread(&copyHost2HostFunction,
                                           iterations_per_thread);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = new int[static_cast<std::size_t>(size)];
    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        array_host[i] = default_value;
    }
    int* array_copy = new int[static_cast<std::size_t>(size)];
    copyHost2HostArray<int>(array_host, size, array_copy, MemoryCopy::NO_CHECK);

    EXPECT_TRUE( thrust::equal(thrust::host,
                               array_host, array_host + size,
                               array_copy,
                               thrust::equal_to<int>()) );

    delete[] array_host;
    delete[] array_copy;
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray_self)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = createHostArray<int>(size, default_value);
    int* array_copy = array_host;
    copyHost2HostArray<int>(array_host, size, array_copy);

    EXPECT_TRUE( thrust::equal(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                               stdgpu::host_cbegin(array_copy),
                               thrust::equal_to<int>()) );

    destroyHostArray<int>(array_host);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, copyHost2HostArray_self_no_check)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = new int[static_cast<std::size_t>(size)];
    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        array_host[i] = default_value;
    }
    int* array_copy = array_host;
    copyHost2HostArray<int>(array_host, size, array_copy, MemoryCopy::NO_CHECK);

    EXPECT_TRUE( thrust::equal(thrust::host,
                               array_host, array_host + size,
                               array_copy,
                               thrust::equal_to<int>()) );

    delete[] array_host;
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyDeviceArray_double_free)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_device   = createDeviceArray<int>(size, default_value);
    int* array_device_2 = array_device;

    destroyDeviceArray<int>(array_device);
    destroyDeviceArray<int>(array_device_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyHostArray_double_free)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host     = createHostArray<int>(size, default_value);
    int* array_host_2   = array_host;

    destroyHostArray<int>(array_host);
    destroyHostArray<int>(array_host_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyManangedArray_double_free)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_managed_device = createManagedArray<int>(size, default_value, Initialization::DEVICE);
    int* array_managed_host = createManagedArray<int>(size, default_value, Initialization::HOST);
    int* array_managed_device_2 = array_managed_device;
    int* array_managed_host_2 = array_managed_host;

    destroyManagedArray<int>(array_managed_device);
    destroyManagedArray<int>(array_managed_device_2);
    destroyManagedArray<int>(array_managed_host);
    destroyManagedArray<int>(array_managed_host_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyDeviceArray_double_free_shifted)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;
    const stdgpu::index_t offset = 24;

    int* array_device = createDeviceArray<int>(size, default_value);
    int* array_device_2 = array_device + offset;

    destroyDeviceArray<int>(array_device);
    destroyDeviceArray<int>(array_device_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyHostArray_double_free_shifted)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;
    const stdgpu::index_t offset = 24;

    int* array_host = createHostArray<int>(size, default_value);
    int* array_host_2 = array_host + offset;

    destroyHostArray<int>(array_host);
    destroyHostArray<int>(array_host_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroyManangedArray_double_free_shifted)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;
    const stdgpu::index_t offset = 24;

    int* array_managed_device = createManagedArray<int>(size, default_value, Initialization::DEVICE);
    int* array_managed_host = createManagedArray<int>(size, default_value, Initialization::HOST);
    int* array_managed_device_2 = array_managed_device + offset;
    int* array_managed_host_2 = array_managed_host + offset;

    destroyManagedArray<int>(array_managed_device);
    destroyManagedArray<int>(array_managed_device_2);
    destroyManagedArray<int>(array_managed_host);
    destroyManagedArray<int>(array_managed_host_2);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, safe_device_allocator)
{
    stdgpu::safe_device_allocator<int> a;
    const stdgpu::index64_t size = 42;

    int* array = a.allocate(size);

    #if STDGPU_DETAIL_IS_DEVICE_COMPILED
        const int default_value = 10;
        thrust::fill(stdgpu::device_begin(array), stdgpu::device_end(array),
                     default_value);

        EXPECT_TRUE( thrust::all_of(stdgpu::device_cbegin(array), stdgpu::device_cend(array),
                                    equal_to_number(default_value)) );
    #endif

    a.deallocate(array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, safe_host_allocator)
{
    stdgpu::safe_host_allocator<int> a;
    const stdgpu::index64_t size = 42;

    int* array = a.allocate(size);

    const int default_value = 10;
    thrust::fill(stdgpu::host_begin(array), stdgpu::host_end(array),
                 default_value);

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                                equal_to_number(default_value)) );


    a.deallocate(array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, safe_managed_allocator)
{
    stdgpu::safe_managed_allocator<int> a;
    const stdgpu::index64_t size = 42;

    int* array = a.allocate(size);

    const int default_value = 10;
    thrust::fill(stdgpu::host_begin(array), stdgpu::host_end(array),
                 default_value);

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                                equal_to_number(default_value)) );


    a.deallocate(array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, allocator_traits_allocate_deallocate)
{
    using Allocator = stdgpu::safe_host_allocator<int>;

    Allocator a;
    const stdgpu::index64_t size = 42;

    int* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    const int default_value = 10;
    thrust::fill(stdgpu::host_begin(array), stdgpu::host_end(array),
                 default_value);

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                                equal_to_number(default_value)) );


    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, allocator_traits_allocate_hint_deallocate)
{
    using Allocator = stdgpu::safe_host_allocator<int>;

    Allocator a;
    const stdgpu::index64_t size = 42;

    int* array_hint = stdgpu::allocator_traits<Allocator>::allocate(a, size);
    int* array = stdgpu::allocator_traits<Allocator>::allocate(a, size, array_hint);

    const int default_value = 10;
    thrust::fill(stdgpu::host_begin(array), stdgpu::host_end(array),
                 default_value);

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array), stdgpu::host_cend(array),
                                equal_to_number(default_value)) );


    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
    stdgpu::allocator_traits<Allocator>::deallocate(a, array_hint, size);
}


namespace
{
    class Counter
    {
        public:
            STDGPU_HOST_DEVICE
            Counter(int* constructor_calls,
                    int* destructor_calls)
                : _constructor_calls(constructor_calls),
                  _destructor_calls(destructor_calls)
            {
                *_constructor_calls += 1;
            }

            STDGPU_HOST_DEVICE
            ~Counter()
            {
                *_destructor_calls += 1;
            }

        private:
            int* _constructor_calls = nullptr;
            int* _destructor_calls = nullptr;
    };


    template <typename Allocator>
    class traits_construct
    {
        public:
            traits_construct(int* constructor_calls,
                             int* destructor_calls)
                : _constructor_calls(constructor_calls),
                  _destructor_calls(destructor_calls)
            {

            }

            void
            operator()(typename Allocator::value_type& value)
            {
                stdgpu::allocator_traits<Allocator>::construct(a,
                                                               &value,
                                                               _constructor_calls,
                                                               _destructor_calls);
            }

        private:
            Allocator a;
            int* _constructor_calls = nullptr;
            int* _destructor_calls = nullptr;
    };


    template <typename Allocator>
    class traits_destroy
    {
        public:
            void
            operator()(typename Allocator::value_type& value)
            {
                stdgpu::allocator_traits<Allocator>::destroy(a,
                                                             &value);
            }

        private:
            Allocator a;
    };
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, allocator_traits_construct_destroy)
{
    using Allocator = stdgpu::safe_host_allocator<Counter>;
    Allocator a;

    const stdgpu::index64_t size = 42;

    typename Allocator::value_type* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    int constructor_calls = 0;
    int destructor_calls = 0;

    thrust::for_each(stdgpu::host_begin(array), stdgpu::host_end(array),
                     traits_construct<Allocator>(&constructor_calls, &destructor_calls));

    EXPECT_EQ(constructor_calls, size);
    EXPECT_EQ(destructor_calls, 0);

    thrust::for_each(stdgpu::host_begin(array), stdgpu::host_end(array),
                     traits_destroy<Allocator>());

    EXPECT_EQ(constructor_calls, size);
    EXPECT_EQ(destructor_calls, size);

    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, allocator_traits_max_size_and_select)
{
    using Allocator = stdgpu::safe_host_allocator<int>;
    Allocator a;

    Allocator b = stdgpu::allocator_traits<Allocator>::select_on_container_copy_construction(a);

    EXPECT_EQ(stdgpu::allocator_traits<Allocator>::max_size(a), stdgpu::allocator_traits<Allocator>::max_size(b));
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, construct_at)
{
    using Allocator = stdgpu::safe_host_allocator<Counter>;
    Allocator a;

    const stdgpu::index64_t size = 42;

    typename Allocator::value_type* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    int constructor_calls = 0;
    int destructor_calls = 0;

    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        stdgpu::construct_at(array + i, &constructor_calls, &destructor_calls);
    }

    EXPECT_EQ(constructor_calls, size);
    EXPECT_EQ(destructor_calls, 0);

    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroy)
{
    using Allocator = stdgpu::safe_host_allocator<Counter>;
    Allocator a;

    const stdgpu::index64_t size = 42;

    typename Allocator::value_type* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    int constructor_calls = 0;
    int destructor_calls = 0;

    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        stdgpu::construct_at(array + i, &constructor_calls, &destructor_calls);
    }

    constructor_calls = 0;
    destructor_calls = 0;
    ASSERT_EQ(constructor_calls, 0);
    ASSERT_EQ(destructor_calls, 0);

    stdgpu::destroy(stdgpu::host_begin(array), stdgpu::host_end(array));

    EXPECT_EQ(constructor_calls, 0);
    EXPECT_EQ(destructor_calls, size);

    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroy_n)
{
    using Allocator = stdgpu::safe_host_allocator<Counter>;
    Allocator a;

    const stdgpu::index64_t size = 42;

    typename Allocator::value_type* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    int constructor_calls = 0;
    int destructor_calls = 0;

    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        stdgpu::construct_at(array + i, &constructor_calls, &destructor_calls);
    }

    constructor_calls = 0;
    destructor_calls = 0;
    ASSERT_EQ(constructor_calls, 0);
    ASSERT_EQ(destructor_calls, 0);

    stdgpu::destroy_n(stdgpu::host_begin(array), size);

    EXPECT_EQ(constructor_calls, 0);
    EXPECT_EQ(destructor_calls, size);

    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, destroy_at)
{
    using Allocator = stdgpu::safe_host_allocator<Counter>;
    Allocator a;

    const stdgpu::index64_t size = 42;

    typename Allocator::value_type* array = stdgpu::allocator_traits<Allocator>::allocate(a, size);

    int constructor_calls = 0;
    int destructor_calls = 0;

    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        stdgpu::construct_at(array + i, &constructor_calls, &destructor_calls);
    }

    constructor_calls = 0;
    destructor_calls = 0;
    ASSERT_EQ(constructor_calls, 0);
    ASSERT_EQ(destructor_calls, 0);

    for (stdgpu::index64_t i = 0; i < size; ++i)
    {
        stdgpu::destroy_at(array + i);
    }

    EXPECT_EQ(constructor_calls, 0);
    EXPECT_EQ(destructor_calls, size);

    stdgpu::allocator_traits<Allocator>::deallocate(a, array, size);
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, safe_pinned_host_allocator)
{
    stdgpu::safe_pinned_host_allocator<int> a;
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = a.allocate(size);

    thrust::fill(stdgpu::host_begin(array_host), stdgpu::host_end(array_host),
                 default_value);

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                equal_to_number(default_value)) );


    a.deallocate(array_host, size);
}


namespace
{
    class default_construct
    {
        public:
            explicit default_construct(const int default_value)
                : _default_value(default_value)
            {

            }

            void
            operator()(int& value) const
            {
                stdgpu::default_allocator_traits::construct(&value, _default_value);
            }

        private:
            int _default_value;
    };


    struct default_destroy
    {
        void
        operator()(int& value) const
        {
            stdgpu::default_allocator_traits::destroy(&value);
        }
    };
}


TEST_F(STDGPU_MEMORY_TEST_CLASS, default_allocator_traits)
{
    const stdgpu::index64_t size = 42;
    const int default_value = 10;

    int* array_host = createHostArray<int>(size);

    thrust::for_each(stdgpu::host_begin(array_host), stdgpu::host_end(array_host),
                     default_construct(default_value));

    EXPECT_TRUE( thrust::all_of(stdgpu::host_cbegin(array_host), stdgpu::host_cend(array_host),
                                equal_to_number(default_value)) );

    thrust::for_each(stdgpu::host_begin(array_host), stdgpu::host_end(array_host),
                     default_destroy());

    destroyHostArray<int>(array_host);
}

