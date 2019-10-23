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
#include <unordered_set>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <stdgpu/bitset.cuh>
#include <stdgpu/iterator.h>
#include <stdgpu/memory.h>



class stdgpu_bitset : public ::testing::Test
{
    protected:
        // Called before each test
        virtual void SetUp()
        {
            bitset_size = 1048576;   // 2^20
            bitset = stdgpu::bitset::createDeviceObject(bitset_size);
        }

        // Called after each test
        virtual void TearDown()
        {
            stdgpu::bitset::destroyDeviceObject(bitset);
        }

        stdgpu::index_t bitset_size;
        stdgpu::bitset bitset;
};



TEST_F(stdgpu_bitset, default_values)
{
    EXPECT_EQ(bitset.count(), 0);
}


struct set_all_bits
{
    stdgpu::bitset bitset;

    set_all_bits(stdgpu::bitset bitset)
        : bitset(bitset)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const stdgpu::index_t i)
    {
        bitset.set(i);

        return bitset[i];
    }
};


struct reset_all_bits
{
    stdgpu::bitset bitset;

    reset_all_bits(stdgpu::bitset bitset)
        : bitset(bitset)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const stdgpu::index_t i)
    {
        bitset.reset(i);

        return bitset[i];
    }
};


struct set_and_reset_all_bits
{
    stdgpu::bitset bitset;

    set_and_reset_all_bits(stdgpu::bitset bitset)
        : bitset(bitset)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const stdgpu::index_t i)
    {
        bitset.set(i);

        if (bitset[i])
        {
            bitset.reset(i);
        }

        return bitset[i];
    }
};


struct flip_all_bits
{
    stdgpu::bitset bitset;

    flip_all_bits(stdgpu::bitset bitset)
        : bitset(bitset)
    {

    }

    STDGPU_DEVICE_ONLY bool
    operator()(const stdgpu::index_t i)
    {
        bitset.flip(i);

        return bitset[i];
    }
};


TEST_F(stdgpu_bitset, set_all_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      set_all_bits(bitset));

    ASSERT_EQ(bitset.count(), bitset.size());


    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);
}


TEST_F(stdgpu_bitset, reset_all_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      set_all_bits(bitset));

    ASSERT_EQ(bitset.count(), bitset.size());

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      reset_all_bits(bitset));

    ASSERT_EQ(bitset.count(), 0);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);
}


TEST_F(stdgpu_bitset, set_and_reset_all_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      set_and_reset_all_bits(bitset));

    ASSERT_EQ(bitset.count(), 0);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);
}


TEST_F(stdgpu_bitset, flip_all_bits_previously_reset)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    // Previously reset
    bitset.reset();

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      flip_all_bits(bitset));

    ASSERT_EQ(bitset.count(), bitset.size());


    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_TRUE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);
}


TEST_F(stdgpu_bitset, flip_all_bits_previously_set)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    // Previously set
    bitset.set();

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(bitset.size()),
                      stdgpu::device_begin(set),
                      flip_all_bits(bitset));

    ASSERT_EQ(bitset.count(), 0);


    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);
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
    std::random_shuffle(host_result, host_result + size);

    return host_result;
}


struct set_bits
{
    stdgpu::bitset bitset;
    stdgpu::index_t* positions;
    uint8_t* set;

    set_bits(stdgpu::bitset bitset,
             stdgpu::index_t* positions,
             uint8_t* set)
        : bitset(bitset),
          positions(positions),
          set(set)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        bitset.set(positions[i]);

        set[positions[i]] = bitset[positions[i]];
    }
};


struct reset_bits
{
    stdgpu::bitset bitset;
    stdgpu::index_t* positions;
    uint8_t* set;

    reset_bits(stdgpu::bitset bitset,
               stdgpu::index_t* positions,
               uint8_t* set)
        : bitset(bitset),
          positions(positions),
          set(set)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        bitset.reset(positions[i]);

        set[positions[i]] = bitset[positions[i]];
    }
};


struct set_and_reset_bits
{
    stdgpu::bitset bitset;
    stdgpu::index_t* positions;
    uint8_t* set;

    set_and_reset_bits(stdgpu::bitset bitset,
                      stdgpu::index_t* positions,
                      uint8_t* set)
        : bitset(bitset),
          positions(positions),
          set(set)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        bitset.set(positions[i]);

        if (bitset[positions[i]])
        {
            bitset.reset(positions[i]);
        }

        set[positions[i]] = bitset[positions[i]];
    }
};


struct flip_bits
{
    stdgpu::bitset bitset;
    stdgpu::index_t* positions;
    uint8_t* set;

    flip_bits(stdgpu::bitset bitset,
              stdgpu::index_t* positions,
              uint8_t* set)
        : bitset(bitset),
          positions(positions),
          set(set)
    {

    }

    STDGPU_DEVICE_ONLY void
    operator()(const stdgpu::index_t i)
    {
        bitset.flip(positions[i]);

        set[positions[i]] = bitset[positions[i]];
    }
};


TEST_F(stdgpu_bitset, set_random_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence  = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence       = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     set_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

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

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}


TEST_F(stdgpu_bitset, reset_random_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence  = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence       = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                      set_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     reset_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), 0);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}


TEST_F(stdgpu_bitset, set_and_reset_random_bits)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size());

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence  = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence       = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     set_and_reset_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), 0);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

    for (stdgpu::index_t i = 0; i < bitset.size(); ++i)
    {
        EXPECT_FALSE(static_cast<bool>(host_set[i]));
    }

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}


TEST_F(stdgpu_bitset, flip_random_bits_previously_reset)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size(), std::numeric_limits<uint8_t>::min());  // Same state as the bitset

    // Previously reset
    bitset.reset();

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence  = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence       = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     flip_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), N);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

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

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}


TEST_F(stdgpu_bitset, flip_random_bits_previously_set)
{
    uint8_t* set = createDeviceArray<uint8_t>(bitset.size(), std::numeric_limits<uint8_t>::max());  // Same state as the bitset

    // Previously set
    bitset.set();

    const stdgpu::index_t N = bitset.size() / 3;
    stdgpu::index_t* host_random_sequence  = generate_shuffled_sequence(bitset.size());
    stdgpu::index_t* random_sequence       = copyCreateHost2DeviceArray<stdgpu::index_t>(host_random_sequence, bitset.size());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                     flip_bits(bitset, random_sequence, set));

    ASSERT_EQ(bitset.count(), bitset.size() - N);

    uint8_t* host_set = copyCreateDevice2HostArray(set, bitset.size());

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

    destroyHostArray<uint8_t>(host_set);
    destroyDeviceArray<uint8_t>(set);

    destroyDeviceArray<stdgpu::index_t>(random_sequence);
    destroyHostArray<stdgpu::index_t>(host_random_sequence);
}


