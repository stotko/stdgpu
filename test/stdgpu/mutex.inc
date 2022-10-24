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

#include <stdgpu/algorithm.h>
#include <stdgpu/functional.h>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/mutex.cuh>
#include <stdgpu/numeric.h>
#include <test_memory_utils.h>

class stdgpu_mutex : public ::testing::Test
{
protected:
    // Called before each test
    void
    SetUp() override
    {
        locks = stdgpu::mutex_array<>::createDeviceObject(locks_size);
    }

    // Called after each test
    void
    TearDown() override
    {
        stdgpu::mutex_array<>::destroyDeviceObject(locks);
    }

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)
    const stdgpu::index_t locks_size = 100000;
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)
    stdgpu::mutex_array<> locks = {};
};

TEST_F(stdgpu_mutex, empty_container)
{
    stdgpu::mutex_array<> empty_container;

    EXPECT_TRUE(empty_container.empty());
    EXPECT_EQ(empty_container.size(), 0);
    EXPECT_TRUE(empty_container.valid());
}

TEST_F(stdgpu_mutex, default_values)
{
    EXPECT_TRUE(locks.valid());
}

class lock_and_unlock
{
public:
    explicit lock_and_unlock(const stdgpu::mutex_array<>& locks)
      : _locks(locks)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        // --- SEQUENTIAL PART ---
        bool leave_loop = false;
        while (!leave_loop)
        {
            if (_locks[i].try_lock())
            {
                // START --- critical section --- START

                // Waste time ...
                long j = 0;
                const int iterations = 10000;
                for (int k = 0; k < iterations; ++k)
                {
                    j += k;
                }
                dummy_j += j;

                //  END  --- critical section ---  END
                leave_loop = true;
                _locks[i].unlock();
            }
        }
        // --- SEQUENTIAL PART ---
    }

private:
    stdgpu::mutex_array<> _locks;
    long dummy_j = 0;
};

TEST_F(stdgpu_mutex, parallel_lock_and_unlock)
{
    stdgpu::for_each_index(stdgpu::execution::device, locks.size(), lock_and_unlock(locks));

    EXPECT_TRUE(locks.valid());
}

class same_state
{
public:
    same_state(const stdgpu::mutex_array<>& locks_1, const stdgpu::mutex_array<>& locks_2, std::uint8_t* equality_flags)
      : _locks_1(locks_1)
      , _locks_2(locks_2)
      , _equality_flags(equality_flags)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _equality_flags[i] = static_cast<std::uint8_t>(_locks_1[i].locked() == _locks_2[i].locked());
    }

private:
    stdgpu::mutex_array<> _locks_1;
    stdgpu::mutex_array<> _locks_2;
    std::uint8_t* _equality_flags;
};

class check_flag
{
public:
    explicit check_flag(std::uint8_t* equality_flags)
      : _equality_flags(equality_flags)
    {
    }

    STDGPU_HOST_DEVICE bool
    operator()(const stdgpu::index_t i) const
    {
        return static_cast<bool>(_equality_flags[i]);
    }

private:
    std::uint8_t* _equality_flags;
};

bool
equal(const stdgpu::mutex_array<>& locks_1, const stdgpu::mutex_array<>& locks_2)
{
    if (locks_1.size() != locks_2.size())
    {
        return false;
    }

    std::uint8_t* equality_flags = createDeviceArray<std::uint8_t>(locks_1.size());

    stdgpu::for_each_index(stdgpu::execution::device, locks_1.size(), same_state(locks_1, locks_2, equality_flags));

    bool result = stdgpu::transform_reduce_index(stdgpu::execution::device,
                                                 locks_1.size(),
                                                 true,
                                                 stdgpu::logical_and<>(),
                                                 check_flag(equality_flags));

    destroyDeviceArray<std::uint8_t>(equality_flags);

    return result;
}

class lock_single_functor
{
public:
    lock_single_functor(const stdgpu::mutex_array<>& locks, const stdgpu::index_t n, std::uint8_t* result)
      : _locks(locks)
      , _n(n)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = static_cast<std::uint8_t>(_locks[_n].try_lock());
    }

private:
    stdgpu::mutex_array<> _locks;
    stdgpu::index_t _n;
    std::uint8_t* _result;
};

bool
lock_single(const stdgpu::mutex_array<>& locks, const stdgpu::index_t n)
{
    std::uint8_t* result = createDeviceArray<std::uint8_t>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, lock_single_functor(locks, n, result));

    std::uint8_t host_result = {};
    copyDevice2HostArray<std::uint8_t>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<std::uint8_t>(result);

    return static_cast<bool>(host_result);
}

TEST_F(stdgpu_mutex, single_try_lock_while_locked)
{
    const stdgpu::index_t n = 42;

    ASSERT_TRUE(lock_single(locks, n));

    stdgpu::mutex_array<> locks_check = stdgpu::mutex_array<>::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n));

    ASSERT_TRUE(equal(locks, locks_check));

    EXPECT_FALSE(lock_single(locks, n));

    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array<>::destroyDeviceObject(locks_check);
}

class lock_multiple_functor
{
public:
    lock_multiple_functor(const stdgpu::mutex_array<>& locks,
                          const stdgpu::index_t n_0,
                          const stdgpu::index_t n_1,
                          int* result)
      : _locks(locks)
      , _n_0(n_0)
      , _n_1(n_1)
      , _result(result)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()([[maybe_unused]] const stdgpu::index_t i)
    {
        *_result = stdgpu::try_lock(_locks[_n_0], _locks[_n_1]);
    }

private:
    stdgpu::mutex_array<> _locks;
    stdgpu::index_t _n_0;
    stdgpu::index_t _n_1;
    int* _result;
};

int
lock_multiple(const stdgpu::mutex_array<>& locks, const stdgpu::index_t n_0, const stdgpu::index_t n_1)
{
    int* result = createDeviceArray<int>(1);

    stdgpu::for_each_index(stdgpu::execution::device, 1, lock_multiple_functor(locks, n_0, n_1, result));

    int host_result = {};
    copyDevice2HostArray<int>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<int>(result);

    return host_result;
}

TEST_F(stdgpu_mutex, multiple_try_lock_both_unlocked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    stdgpu::mutex_array<> locks_check = stdgpu::mutex_array<>::createDeviceObject(locks_size);

    ASSERT_TRUE(equal(locks, locks_check));

    EXPECT_EQ(lock_multiple(locks, n_0, n_1), -1);

    // Both mutexes should be locked now
    ASSERT_TRUE(lock_single(locks_check, n_0));
    ASSERT_TRUE(lock_single(locks_check, n_1));
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array<>::destroyDeviceObject(locks_check);
}

TEST_F(stdgpu_mutex, multiple_try_lock_first_unlocked_second_locked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_1));

    stdgpu::mutex_array<> locks_check = stdgpu::mutex_array<>::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));

    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 1);

    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array<>::destroyDeviceObject(locks_check);
}

TEST_F(stdgpu_mutex, multiple_try_lock_first_locked_second_unlocked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_0));

    stdgpu::mutex_array<> locks_check = stdgpu::mutex_array<>::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_0));

    ASSERT_TRUE(equal(locks, locks_check));

    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 0);

    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array<>::destroyDeviceObject(locks_check);
}

TEST_F(stdgpu_mutex, multiple_try_lock_both_locked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_0));
    ASSERT_TRUE(lock_single(locks, n_1));

    stdgpu::mutex_array<> locks_check = stdgpu::mutex_array<>::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_0));
    ASSERT_TRUE(lock_single(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));

    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 0);

    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array<>::destroyDeviceObject(locks_check);
}

TEST_F(stdgpu_mutex, get_allocator)
{
    const stdgpu::index_t N = 10000;

    stdgpu::mutex_array<> lock2 = stdgpu::mutex_array<>::createDeviceObject(N);

    stdgpu::mutex_array<>::allocator_type a = lock2.get_allocator();

    stdgpu::mutex_default_type* array = a.allocate(N);
    a.deallocate(array, N);

    stdgpu::mutex_array<>::destroyDeviceObject(lock2);
}

TEST_F(stdgpu_mutex, custom_allocator)
{
    test_utils::get_allocator_statistics().reset();

    {
        const stdgpu::index_t N = 10000;

        using Allocator = test_utils::test_device_allocator<stdgpu::mutex_default_type>;
        Allocator a_orig;

        stdgpu::mutex_array<stdgpu::mutex_default_type, Allocator> lock2 =
                stdgpu::mutex_array<stdgpu::mutex_default_type, Allocator>::createDeviceObject(N, a_orig);

        stdgpu::mutex_array<stdgpu::mutex_default_type, Allocator>::allocator_type a = lock2.get_allocator();

        stdgpu::mutex_default_type* array = a.allocate(N);
        a.deallocate(array, N);

        stdgpu::mutex_array<stdgpu::mutex_default_type, Allocator>::destroyDeviceObject(lock2);
    }

    // Account for potential but not guaranteed copy-ellision
    EXPECT_EQ(test_utils::get_allocator_statistics().default_constructions, 1);
    EXPECT_GE(test_utils::get_allocator_statistics().copy_constructions, 3);
    EXPECT_LE(test_utils::get_allocator_statistics().copy_constructions, 5);
    EXPECT_GE(test_utils::get_allocator_statistics().destructions, 4);
    EXPECT_LE(test_utils::get_allocator_statistics().destructions, 6);

    test_utils::get_allocator_statistics().reset();
}
