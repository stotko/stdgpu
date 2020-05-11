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

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>

#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <stdgpu/mutex.cuh>



class stdgpu_mutex : public ::testing::Test
{
    protected:
        // Called before each test
        void SetUp() override
        {
            locks = stdgpu::mutex_array::createDeviceObject(locks_size);
        }

        // Called after each test
        void TearDown() override
        {
            stdgpu::mutex_array::destroyDeviceObject(locks);
        }

        const stdgpu::index_t locks_size = 100000; // NOLINT(misc-non-private-member-variables-in-classes)
        stdgpu::mutex_array locks = {}; // NOLINT(misc-non-private-member-variables-in-classes)
};



TEST_F(stdgpu_mutex, default_values)
{
    EXPECT_TRUE(locks.valid());
}


class lock_and_unlock
{
    public:
        explicit lock_and_unlock(const stdgpu::mutex_array& locks)
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

                    //  END  --- critical section ---  END
                    leave_loop = true;
                    _locks[i].unlock();
                }
            }
            // --- SEQUENTIAL PART ---
        }

    private:
        stdgpu::mutex_array _locks;
};


TEST_F(stdgpu_mutex, parallel_lock_and_unlock)
{
    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(locks.size()),
                     lock_and_unlock(locks));

    EXPECT_TRUE(locks.valid());
}


class same_state
{
    public:
        same_state(const stdgpu::mutex_array& locks_1,
                   const stdgpu::mutex_array& locks_2)
            : _locks_1(locks_1),
              _locks_2(locks_2)
        {

        }

        STDGPU_DEVICE_ONLY bool
        operator()(const stdgpu::index_t i)
        {
            return _locks_1[i].locked() == _locks_2[i].locked();
        }

    private:
        stdgpu::mutex_array _locks_1;
        stdgpu::mutex_array _locks_2;
};


struct check_flag
{
    STDGPU_DEVICE_ONLY bool
    operator()(const std::uint8_t flag) const
    {
        return static_cast<bool>(flag);
    }
};


bool
equal(const stdgpu::mutex_array& locks_1,
      const stdgpu::mutex_array& locks_2)
{
    if (locks_1.size() != locks_2.size())
    {
        return false;
    }

    std::uint8_t* equality_flags = createDeviceArray<std::uint8_t>(locks_1.size());

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(locks_1.size()),
                      stdgpu::device_begin(equality_flags),
                      same_state(locks_1, locks_2));

    bool result = thrust::all_of(stdgpu::device_cbegin(equality_flags), stdgpu::device_cend(equality_flags),
                                 check_flag());

    destroyDeviceArray<std::uint8_t>(equality_flags);

    return result;
}


class lock_single_functor
{
    public:
        explicit lock_single_functor(const stdgpu::mutex_array& locks)
            : _locks(locks)
        {

        }

        STDGPU_DEVICE_ONLY bool
        operator()(const stdgpu::index_t i)
        {
            return _locks[i].try_lock();
        }

    private:
        stdgpu::mutex_array _locks;
};


bool
lock_single(const stdgpu::mutex_array& locks,
            const stdgpu::index_t n)
{
    std::uint8_t* result = createDeviceArray<std::uint8_t>(1);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(n), thrust::counting_iterator<stdgpu::index_t>(n + 1),
                      stdgpu::device_begin(result),
                      lock_single_functor(locks));

    std::uint8_t host_result;
    copyDevice2HostArray<std::uint8_t>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<std::uint8_t>(result);

    return static_cast<bool>(host_result);
}


TEST_F(stdgpu_mutex, single_try_lock_while_locked)
{
    const stdgpu::index_t n = 42;

    ASSERT_TRUE(lock_single(locks, n));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_FALSE(lock_single(locks, n));


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


class lock_multiple_functor
{
    public:
        explicit lock_multiple_functor(const stdgpu::mutex_array& locks)
            : _locks(locks)
        {

        }

        STDGPU_DEVICE_ONLY int
        operator()(const thrust::tuple<stdgpu::index_t, stdgpu::index_t> i)
        {
            return stdgpu::try_lock(_locks[thrust::get<0>(i)], _locks[thrust::get<1>(i)]);
        }

    private:
        stdgpu::mutex_array _locks;
};

int
lock_multiple(const stdgpu::mutex_array& locks,
              const stdgpu::index_t n_0,
              const stdgpu::index_t n_1)
{
    int* result = createDeviceArray<int>(1);

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<stdgpu::index_t>(n_0),     thrust::counting_iterator<stdgpu::index_t>(n_1))),
                      thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<stdgpu::index_t>(n_0 + 1), thrust::counting_iterator<stdgpu::index_t>(n_1 + 1))),
                      stdgpu::device_begin(result),
                      lock_multiple_functor(locks));

    int host_result;
    copyDevice2HostArray<int>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<int>(result);

    return host_result;
}


TEST_F(stdgpu_mutex, multiple_try_lock_both_unlocked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple(locks, n_0, n_1), -1);


    // Both mutexes should be locked now
    ASSERT_TRUE(lock_single(locks_check, n_0));
    ASSERT_TRUE(lock_single(locks_check, n_1));
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_first_unlocked_second_locked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_1));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 1);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_first_locked_second_unlocked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_0));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_0));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 0);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_both_locked)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single(locks, n_0));
    ASSERT_TRUE(lock_single(locks, n_1));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single(locks_check, n_0));
    ASSERT_TRUE(lock_single(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple(locks, n_0, n_1), 0);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


class lock_multiple_functor_new_reference
{
    public:
        explicit lock_multiple_functor_new_reference(const stdgpu::mutex_array& locks)
            : _locks(locks)
        {

        }

        STDGPU_DEVICE_ONLY int
        operator()(const thrust::tuple<stdgpu::index_t, stdgpu::index_t> i)
        {
            stdgpu::mutex_array::reference ref_0 = static_cast<stdgpu::mutex_array::reference>(_locks[thrust::get<0>(i)]);
            stdgpu::mutex_array::reference ref_1 = static_cast<stdgpu::mutex_array::reference>(_locks[thrust::get<1>(i)]);
            return stdgpu::try_lock(ref_0, ref_1);
        }

    private:
        stdgpu::mutex_array _locks;
};

int
lock_multiple_new_reference(const stdgpu::mutex_array& locks,
                            const stdgpu::index_t n_0,
                            const stdgpu::index_t n_1)
{
    int* result = createDeviceArray<int>(1);

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<stdgpu::index_t>(n_0),     thrust::counting_iterator<stdgpu::index_t>(n_1))),
                      thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<stdgpu::index_t>(n_0 + 1), thrust::counting_iterator<stdgpu::index_t>(n_1 + 1))),
                      stdgpu::device_begin(result),
                      lock_multiple_functor_new_reference(locks));

    int host_result;
    copyDevice2HostArray<int>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<int>(result);

    return host_result;
}


class lock_single_functor_new_reference
{
    public:
        explicit lock_single_functor_new_reference(const stdgpu::mutex_array& locks)
            : _locks(locks)
        {

        }

        STDGPU_DEVICE_ONLY bool
        operator()(const stdgpu::index_t i)
        {
            stdgpu::mutex_array::reference ref = static_cast<stdgpu::mutex_array::reference>(_locks[i]);
            return ref.try_lock();
        }

    private:
        stdgpu::mutex_array _locks;
};

bool
lock_single_new_reference(const stdgpu::mutex_array& locks,
                          const stdgpu::index_t n)
{
    std::uint8_t* result = createDeviceArray<std::uint8_t>(1);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(n), thrust::counting_iterator<stdgpu::index_t>(n + 1),
                      stdgpu::device_begin(result),
                      lock_single_functor_new_reference(locks));

    std::uint8_t host_result;
    copyDevice2HostArray<std::uint8_t>(result, 1, &host_result, MemoryCopy::NO_CHECK);

    destroyDeviceArray<std::uint8_t>(result);

    return static_cast<bool>(host_result);
}


TEST_F(stdgpu_mutex, multiple_try_lock_both_unlocked_new_reference)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple_new_reference(locks, n_0, n_1), -1);


    // Both mutexes should be locked now
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_0));
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_1));
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_first_unlocked_second_locked_new_reference)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single_new_reference(locks, n_1));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple_new_reference(locks, n_0, n_1), 1);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_first_locked_second_unlocked_new_reference)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single_new_reference(locks, n_0));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_0));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple_new_reference(locks, n_0, n_1), 0);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


TEST_F(stdgpu_mutex, multiple_try_lock_both_locked_new_reference)
{
    const stdgpu::index_t n_0 = 21;
    const stdgpu::index_t n_1 = 42;

    ASSERT_TRUE(lock_single_new_reference(locks, n_0));
    ASSERT_TRUE(lock_single_new_reference(locks, n_1));

    stdgpu::mutex_array locks_check = stdgpu::mutex_array::createDeviceObject(locks_size);
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_0));
    ASSERT_TRUE(lock_single_new_reference(locks_check, n_1));

    ASSERT_TRUE(equal(locks, locks_check));


    EXPECT_EQ(lock_multiple_new_reference(locks, n_0, n_1), 0);


    // Nothing has changed
    EXPECT_TRUE(equal(locks, locks_check));

    stdgpu::mutex_array::destroyDeviceObject(locks_check);
}


