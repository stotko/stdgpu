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

#include <algorithm>
#include <limits>
#include <random>
#include <unordered_set>

#include <stdgpu/algorithm.h>
#include <stdgpu/bitset.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>
#include <test_memory_utils.h>
#include <test_utils.h>

class stdgpu_bitset : public ::testing::Test
{
protected:
    // Called before each test
    void
    SetUp() override
    {
        bitset = stdgpu::bitset<>::createDeviceObject(bitset_size);
    }

    // Called after each test
    void
    TearDown() override
    {
        stdgpu::bitset<>::destroyDeviceObject(bitset);
    }

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)
    const stdgpu::index_t bitset_size = 1048576;
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)
    stdgpu::bitset<> bitset = {};
};

TEST_F(stdgpu_bitset, empty_container)
{
    stdgpu::bitset<> empty_container;

    EXPECT_TRUE(empty_container.empty());
    EXPECT_EQ(empty_container.size(), 0);
    EXPECT_EQ(empty_container.count(), 0);
    EXPECT_FALSE(empty_container.all());
    EXPECT_FALSE(empty_container.any());
    EXPECT_FALSE(empty_container.none());
}

TEST_F(stdgpu_bitset, default_values)
{
    EXPECT_EQ(bitset.count(), 0);
    EXPECT_FALSE(bitset.all());
    EXPECT_FALSE(bitset.any());
    EXPECT_TRUE(bitset.none());
}

class set_all_bits
{
public:
    set_all_bits(const stdgpu::bitset<>& bitset, std::uint8_t* set)
      : _bitset(bitset)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.set(i);

        // Test both access operators at the same time
        _set[i] = static_cast<std::uint8_t>(_bitset[i] && _bitset.test(i));
    }

private:
    stdgpu::bitset<> _bitset;
    std::uint8_t* _set;
};

class reset_all_bits
{
public:
    reset_all_bits(const stdgpu::bitset<>& bitset, std::uint8_t* set)
      : _bitset(bitset)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.reset(i);

        // Test both access operators at the same time
        _set[i] = static_cast<std::uint8_t>(_bitset[i] && _bitset.test(i));
    }

private:
    stdgpu::bitset<> _bitset;
    std::uint8_t* _set;
};

class set_and_reset_all_bits
{
public:
    set_and_reset_all_bits(const stdgpu::bitset<>& bitset, std::uint8_t* set)
      : _bitset(bitset)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.set(i);

        if (_bitset[i])
        {
            _bitset.reset(i);
        }

        // Test both access operators at the same time
        _set[i] = static_cast<std::uint8_t>(_bitset[i] && _bitset.test(i));
    }

private:
    stdgpu::bitset<> _bitset;
    std::uint8_t* _set;
};

class flip_all_bits
{
public:
    flip_all_bits(const stdgpu::bitset<>& bitset, std::uint8_t* set)
      : _bitset(bitset)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.flip(i);

        // Test both access operators at the same time
        _set[i] = static_cast<std::uint8_t>(_bitset[i] && _bitset.test(i));
    }

private:
    stdgpu::bitset<> _bitset;
    std::uint8_t* _set;
};

TEST_F(stdgpu_bitset, set_all_bits_componentwise)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), set_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), bitset.size());

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, reset_all_bits_componentwise)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), set_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), reset_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, set_and_reset_all_bits_componentwise)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), set_and_reset_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, flip_all_bits_previously_reset_componentwise)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    // Previously reset
    bitset.reset();

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), flip_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), bitset.size());

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, flip_all_bits_previously_set_componentwise)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    // Previously set
    bitset.set();

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), flip_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

class read_all_bits
{
public:
    read_all_bits(const stdgpu::bitset<>& bitset, std::uint8_t* set)
      : _bitset(bitset)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        // Test both access operators at the same time
        _set[i] = static_cast<std::uint8_t>(_bitset[i] && _bitset.test(i));
    }

private:
    stdgpu::bitset<> _bitset;
    std::uint8_t* _set;
};

TEST_F(stdgpu_bitset, set_all_bits)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    bitset.set();

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), read_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), bitset.size());

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, reset_all_bits)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    bitset.set();

    ASSERT_EQ(bitset.count(), bitset.size());

    bitset.reset();

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), read_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, flip_all_bits_previously_reset)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    // Previously reset
    bitset.reset();

    bitset.flip();

    stdgpu::for_each_index(stdgpu::execution::device, bitset.size(), read_all_bits(bitset, set));

    ASSERT_EQ(bitset.count(), bitset.size());

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

TEST_F(stdgpu_bitset, flip_all_bits_previously_set)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    // Previously set
    bitset.set();

    bitset.flip();

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);
}

stdgpu::index_t*
generate_shuffled_sequence(const stdgpu::index_t size)
{
    stdgpu::index_t* host_result = createHostArray<stdgpu::index_t>(size);

    // Sequence
    for (stdgpu::index_t i = 0; i < size; ++i)
    {
        host_result[i] = i;
    }

    // Shuffle
    std::size_t seed = test_utils::random_thread_seed();
    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));

    std::shuffle(host_result, host_result + size, rng);

    return host_result;
}

class set_bits
{
public:
    set_bits(const stdgpu::bitset<>& bitset, stdgpu::index_t* positions, std::uint8_t* set)
      : _bitset(bitset)
      , _positions(positions)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.set(_positions[i]);

        _set[_positions[i]] = static_cast<std::uint8_t>(_bitset[_positions[i]]);
    }

private:
    stdgpu::bitset<> _bitset;
    stdgpu::index_t* _positions;
    std::uint8_t* _set;
};

class reset_bits
{
public:
    reset_bits(const stdgpu::bitset<>& bitset, stdgpu::index_t* positions, std::uint8_t* set)
      : _bitset(bitset)
      , _positions(positions)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.reset(_positions[i]);

        _set[_positions[i]] = static_cast<std::uint8_t>(_bitset[_positions[i]]);
    }

private:
    stdgpu::bitset<> _bitset;
    stdgpu::index_t* _positions;
    std::uint8_t* _set;
};

class set_and_reset_bits
{
public:
    set_and_reset_bits(const stdgpu::bitset<>& bitset, stdgpu::index_t* positions, std::uint8_t* set)
      : _bitset(bitset)
      , _positions(positions)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.set(_positions[i]);

        if (_bitset[_positions[i]])
        {
            _bitset.reset(_positions[i]);
        }

        _set[_positions[i]] = static_cast<std::uint8_t>(_bitset[_positions[i]]);
    }

private:
    stdgpu::bitset<> _bitset;
    stdgpu::index_t* _positions;
    std::uint8_t* _set;
};

class flip_bits
{
public:
    flip_bits(const stdgpu::bitset<>& bitset, stdgpu::index_t* positions, std::uint8_t* set)
      : _bitset(bitset)
      , _positions(positions)
      , _set(set)
    {
    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        _bitset.flip(_positions[i]);

        _set[_positions[i]] = static_cast<std::uint8_t>(_bitset[_positions[i]]);
    }

private:
    stdgpu::bitset<> _bitset;
    stdgpu::index_t* _positions;
    std::uint8_t* _set;
};

TEST_F(stdgpu_bitset, set_random_bits)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, N, set_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    std::unordered_set<stdgpu::index_t> host_random_sequence_container(host_random_sequence, host_random_sequence + N);
    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        if (host_random_sequence_container.find(i) != host_random_sequence_container.end())
        {
            EXPECT_TRUE(static_cast<bool>(host_set[i]));
        }
        else
        {
            EXPECT_FALSE(static_cast<bool>(host_set[i]));
        }
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}

TEST_F(stdgpu_bitset, reset_random_bits)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, N, set_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    stdgpu::for_each_index(stdgpu::execution::device, N, reset_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}

TEST_F(stdgpu_bitset, set_and_reset_random_bits)
{
    std::uint8_t* set = createDeviceArray<std::uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, N, set_and_reset_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), 0);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}

TEST_F(stdgpu_bitset, flip_random_bits_previously_reset)
{
    std::uint8_t* set =
            createDeviceArray<std::uint8_t>(bitset.size(),
                                            std::numeric_limits<std::uint8_t>::min()); // Same state as the bitset

    // Previously reset
    bitset.reset();

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, N, flip_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    std::unordered_set<stdgpu::index_t> host_random_sequence_container(host_random_sequence, host_random_sequence + N);
    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        if (host_random_sequence_container.find(i) != host_random_sequence_container.end())
        {
            EXPECT_TRUE(static_cast<bool>(host_set[i]));
        }
        else
        {
            EXPECT_FALSE(static_cast<bool>(host_set[i]));
        }
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}

TEST_F(stdgpu_bitset, flip_random_bits_previously_set)
{
    std::uint8_t* set =
            createDeviceArray<std::uint8_t>(bitset.size(),
                                            std::numeric_limits<std::uint8_t>::max()); // Same state as the bitset

    // Previously set
    bitset.set();

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    stdgpu::for_each_index(stdgpu::execution::device, N, flip_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), bitset.size() - N);

    std::uint8_t* host_set = copyCreateDevice2HostArray<std::uint8_t>(set, bitset.size());

    std::unordered_set<stdgpu::index_t> host_random_sequence_container(host_random_sequence, host_random_sequence + N);
    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        if (host_random_sequence_container.find(i) != host_random_sequence_container.end())
        {
            EXPECT_FALSE(static_cast<bool>(host_set[i]));
        }
        else
        {
            EXPECT_TRUE(static_cast<bool>(host_set[i]));
        }
    }

    destroyHostArray<std::uint8_t>(host_set);
    destroyDeviceArray<std::uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}

TEST_F(stdgpu_bitset, get_allocator)
{
    const stdgpu::index_t N = 10000;

    stdgpu::bitset<> bits = stdgpu::bitset<>::createDeviceObject(N);

    stdgpu::bitset<>::allocator_type a = bits.get_allocator();

    stdgpu::bitset_default_type* array = a.allocate(N);
    a.deallocate(array, N);

    stdgpu::bitset<>::destroyDeviceObject(bits);
}

TEST_F(stdgpu_bitset, custom_allocator)
{
    test_utils::get_allocator_statistics().reset();

    {
        const stdgpu::index_t N = 10000;

        using Allocator = test_utils::test_device_allocator<stdgpu::bitset_default_type>;
        Allocator a_orig;

        stdgpu::bitset<stdgpu::bitset_default_type, Allocator> bits =
                stdgpu::bitset<stdgpu::bitset_default_type, Allocator>::createDeviceObject(N, a_orig);

        stdgpu::bitset<stdgpu::bitset_default_type, Allocator>::allocator_type a = bits.get_allocator();

        stdgpu::bitset_default_type* array = a.allocate(N);
        a.deallocate(array, N);

        stdgpu::bitset<stdgpu::bitset_default_type, Allocator>::destroyDeviceObject(bits);
    }

    // Account for potential but not guaranteed copy-ellision
    EXPECT_EQ(test_utils::get_allocator_statistics().default_constructions, 1);
    EXPECT_GE(test_utils::get_allocator_statistics().copy_constructions, 2);
    EXPECT_LE(test_utils::get_allocator_statistics().copy_constructions, 3);
    EXPECT_GE(test_utils::get_allocator_statistics().destructions, 3);
    EXPECT_LE(test_utils::get_allocator_statistics().destructions, 4);

    test_utils::get_allocator_statistics().reset();
}
