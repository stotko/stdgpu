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

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS
    #error "Class name for unit test not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_TYPE
    #error "Data structure type not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE
    #error "Key to Value conversion not specified!"
#endif

#ifndef STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY
    #error "Value to Key conversion not specified!"
#endif



#include <gtest/gtest.h>

#include <random>
#include <thread>
#include <unordered_set>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <test_utils.h>
#include <stdgpu/memory.h>
#include <stdgpu/vector.cuh>



// convenience wrapper to improve readability
using test_unordered_datastructure = STDGPU_UNORDERED_DATASTRUCTURE_TYPE;



class STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS : public ::testing::Test
{
    protected:
        // Called before each test
        void SetUp() override
        {
            hash_datastructure = test_unordered_datastructure::createDeviceObject(hash_datastructure_size);
        }

        // Called after each test
        void TearDown() override
        {
            test_unordered_datastructure::destroyDeviceObject(hash_datastructure);
        }

        const stdgpu::index_t hash_datastructure_size = 100000; // NOLINT(misc-non-private-member-variables-in-classes)
        test_unordered_datastructure hash_datastructure = {}; // NOLINT(misc-non-private-member-variables-in-classes)
};



namespace
{
    void
    thread_hash_inside_range(const stdgpu::index_t iterations,
                             const test_unordered_datastructure& hash_datastructure)
    {
        // Generate true random numbers
        size_t seed = test_utils::random_thread_seed();

        std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
        std::uniform_int_distribution<std::int16_t> dist(std::numeric_limits<std::int16_t>::lowest(), std::numeric_limits<std::int16_t>::max());

        for (stdgpu::index_t i = 0; i < iterations; ++i)
        {
            test_unordered_datastructure::key_type random(dist(rng), dist(rng), dist(rng));

            EXPECT_LT(hash_datastructure.bucket(random), hash_datastructure.bucket_count());
        }
    }
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, hash_inside_range)
{
    const stdgpu::index_t iterations_per_thread = static_cast<stdgpu::index_t>(pow(2, 21));

    test_utils::for_each_concurrent_thread(&thread_hash_inside_range,
                                           iterations_per_thread,
                                           hash_datastructure);
}


namespace
{
    class random_key
    {
        public:
            STDGPU_HOST_DEVICE
            explicit random_key(const std::size_t seed)
                : _seed(seed)
            {

            }

            STDGPU_HOST_DEVICE test_unordered_datastructure::key_type
            operator()(const stdgpu::index_t n) const
            {
                thrust::default_random_engine rng(static_cast<thrust::default_random_engine::result_type>(_seed));
                thrust::uniform_real_distribution<std::int16_t> dist(stdgpu::numeric_limits<std::int16_t>::min(), stdgpu::numeric_limits<std::int16_t>::max());
                rng.discard(static_cast<unsigned long long int>(3) * static_cast<unsigned long long int>(n));

                return test_unordered_datastructure::key_type(dist(rng), dist(rng), dist(rng));
            }

        private:
            std::size_t _seed;
    };


    class count_buckets_hits
    {
        public:
            count_buckets_hits(const test_unordered_datastructure& hash_datastructure,
                               int* bucket_hits)
                : _hash_datastructure(hash_datastructure),
                  _bucket_hits(bucket_hits)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const test_unordered_datastructure::key_type key)
            {
                stdgpu::index_t bucket = _hash_datastructure.bucket(key);

                stdgpu::atomic_ref<int>(_bucket_hits[bucket]).fetch_add(1);
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            int* _bucket_hits;
    };


    template <int threshold>
    struct greater_value
    {
        STDGPU_HOST_DEVICE
        bool operator()(const int number) const
        {
            return (number > threshold);
        }
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, bucket_number_hits)
{
    int* bucket_hits = createDeviceArray<int>(hash_datastructure.bucket_count(), 0);

    // Use more samples than buckets to test how well they are distributed in general
    const stdgpu::index_t N = 2 * hash_datastructure.bucket_count();
    test_unordered_datastructure::key_type* keys = createDeviceArray<test_unordered_datastructure::key_type>(N);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                      stdgpu::device_begin(keys),
                      random_key(test_utils::random_seed()));

    thrust::for_each(stdgpu::device_begin(keys), stdgpu::device_end(keys),
                     count_buckets_hits(hash_datastructure, bucket_hits));


    // Number of saved hash values correct
    stdgpu::index_t number_hash_values = thrust::reduce(stdgpu::device_cbegin(bucket_hits), stdgpu::device_cend(bucket_hits),
                                                        0,
                                                        thrust::plus<int>());

    EXPECT_EQ(number_hash_values, N);



    // Number of hits (buckets with > 0 elements)
    stdgpu::index_t number_hits = static_cast<stdgpu::index_t>(thrust::count_if(stdgpu::device_cbegin(bucket_hits), stdgpu::device_cend(bucket_hits),
                                                               greater_value<0>()));

    const float percent_hits = 80.0F;
    EXPECT_GT(number_hits, static_cast<stdgpu::index_t>(static_cast<float>(hash_datastructure.bucket_count()) * percent_hits / 100.0F));

    destroyDeviceArray<int>(bucket_hits);
    destroyDeviceArray<test_unordered_datastructure::key_type>(keys);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, bucket_number_collisions)
{
    int* bucket_hits = createDeviceArray<int>(hash_datastructure.bucket_count(), 0);

    // Use as many samples as buckets to test how many collisions in percent may appear in general
    const stdgpu::index_t N = hash_datastructure.bucket_count();
    test_unordered_datastructure::key_type* keys = createDeviceArray<test_unordered_datastructure::key_type>(N);

    thrust::transform(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                      stdgpu::device_begin(keys),
                      random_key(test_utils::random_seed()));

    thrust::for_each(stdgpu::device_begin(keys), stdgpu::device_end(keys),
                     count_buckets_hits(hash_datastructure, bucket_hits));


    // Number of saved hash values correct
    stdgpu::index_t number_hash_values = thrust::reduce(stdgpu::device_cbegin(bucket_hits), stdgpu::device_cend(bucket_hits),
                                                        0,
                                                        thrust::plus<int>());

    EXPECT_EQ(number_hash_values, N);



    // Number of collisions (buckets with > 1 elements)
    stdgpu::index_t number_collisions = static_cast<stdgpu::index_t>(thrust::count_if(stdgpu::device_cbegin(bucket_hits), stdgpu::device_cend(bucket_hits),
                                                                     greater_value<1>()));

    const float percent_collisions = 40.0F;
    EXPECT_LT(number_collisions, static_cast<stdgpu::index_t>(static_cast<float>(N) * percent_collisions / 100.0F));

    destroyDeviceArray<int>(bucket_hits);
    destroyDeviceArray<test_unordered_datastructure::key_type>(keys);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, empty_size_limits)
{
    EXPECT_LE(hash_datastructure.size(), hash_datastructure.max_size());
    EXPECT_LE(hash_datastructure.load_factor(), hash_datastructure.max_load_factor());
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, hash_objects)
{
    test_unordered_datastructure::key_equal key_equals  = hash_datastructure.key_eq();
    test_unordered_datastructure::hasher hash           = hash_datastructure.hash_function();

    test_unordered_datastructure::key_type key = random_key(test_utils::random_seed())(0);

    std::size_t key_hash_1 = hash(key);
    std::size_t key_hash_2 = hash(key);

    EXPECT_TRUE(key_equals(key, key));
    EXPECT_EQ(key_hash_1, key_hash_2);
}


namespace
{
    class insert_single
    {
        public:
            insert_single(const test_unordered_datastructure& hash_datastructure,
                          const test_unordered_datastructure::key_type& key,
                          stdgpu::index_t* inserted)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _inserted(inserted)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                thrust::pair<test_unordered_datastructure::iterator, bool> success = _hash_datastructure.insert(STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_key));

                *_inserted = success.second ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            stdgpu::index_t* _inserted;
    };


    bool
    insert_key(test_unordered_datastructure& hash_datastructure,
               const test_unordered_datastructure::key_type& key)
    {
        stdgpu::index_t* inserted = createDeviceArray<stdgpu::index_t>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         insert_single(hash_datastructure, key, inserted));

        stdgpu::index_t host_inserted;
        copyDevice2HostArray<stdgpu::index_t>(inserted, 1, &host_inserted, MemoryCopy::NO_CHECK);

        destroyDeviceArray<stdgpu::index_t>(inserted);

        return host_inserted == 1;
    }


    class erase_single
    {
        public:
            erase_single(const test_unordered_datastructure& hash_datastructure,
                         const test_unordered_datastructure::key_type& key,
                         stdgpu::index_t* erased)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _erased(erased)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_erased = _hash_datastructure.erase(_key);
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            stdgpu::index_t* _erased;
    };


    bool
    erase_key(test_unordered_datastructure& hash_datastructure,
              const test_unordered_datastructure::key_type& key)
    {
        stdgpu::index_t* erased = createDeviceArray<stdgpu::index_t>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         erase_single(hash_datastructure, key, erased));

        stdgpu::index_t host_erased;
        copyDevice2HostArray<stdgpu::index_t>(erased, 1, &host_erased, MemoryCopy::NO_CHECK);

        destroyDeviceArray<stdgpu::index_t>(erased);

        return host_erased == 1;
    }


    class contains_key_functor
    {
        public:
            contains_key_functor(const test_unordered_datastructure& hash_datastructure,
                                 const test_unordered_datastructure::key_type& key,
                                 stdgpu::index_t* contained)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _contained(contained)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_contained = _hash_datastructure.contains(_key) ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            stdgpu::index_t* _contained;
    };


    bool
    contains_key(const test_unordered_datastructure& hash_datastructure,
                 const test_unordered_datastructure::key_type& key)
    {
        stdgpu::index_t* contained = createDeviceArray<stdgpu::index_t>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         contains_key_functor(hash_datastructure, key, contained));

        stdgpu::index_t host_contained;
        copyDevice2HostArray<stdgpu::index_t>(contained, 1, &host_contained, MemoryCopy::NO_CHECK);

        destroyDeviceArray<stdgpu::index_t>(contained);

        return host_contained == 1;
    }


    class non_const_find_key_functor
    {
        public:
            non_const_find_key_functor(const test_unordered_datastructure& hash_datastructure,
                                       const test_unordered_datastructure::key_type& key,
                                       test_unordered_datastructure::iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.find(_key);
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            test_unordered_datastructure::iterator* _result;
    };


    class const_find_key_functor
    {
        public:
            const_find_key_functor(const test_unordered_datastructure& hash_datastructure,
                                   const test_unordered_datastructure::key_type& key,
                                   test_unordered_datastructure::const_iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.find(_key);
            }

        private:
            const test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            test_unordered_datastructure::const_iterator* _result;
    };


    test_unordered_datastructure::iterator
    non_const_find_key(const test_unordered_datastructure& hash_datastructure,
                       const test_unordered_datastructure::key_type& key)
    {
        test_unordered_datastructure::iterator* result = createDeviceArray<test_unordered_datastructure::iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         non_const_find_key_functor(hash_datastructure, key, result));

        test_unordered_datastructure::iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    const_find_key(const test_unordered_datastructure& hash_datastructure,
                   const test_unordered_datastructure::key_type& key)
    {
        test_unordered_datastructure::const_iterator* result = createDeviceArray<test_unordered_datastructure::const_iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         const_find_key_functor(hash_datastructure, key, result));

        test_unordered_datastructure::const_iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::const_iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::const_iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    find_key(const test_unordered_datastructure& hash_datastructure,
             const test_unordered_datastructure::key_type& key)
    {
        test_unordered_datastructure::iterator non_const_iterator   = non_const_find_key(hash_datastructure, key);
        test_unordered_datastructure::const_iterator const_iterator = const_find_key(hash_datastructure, key);

        EXPECT_EQ(non_const_iterator, const_iterator);

        return const_iterator;
    }


    class non_const_begin_iterator_functor
    {
        public:
            non_const_begin_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                             test_unordered_datastructure::iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.begin();
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::iterator* _result;
    };


    class const_begin_iterator_functor
    {
        public:
            const_begin_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                         test_unordered_datastructure::const_iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.begin();
            }

        private:
            const test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::const_iterator* _result;
    };


    class cbegin_iterator_functor
    {
        public:
            cbegin_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                    test_unordered_datastructure::const_iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.cbegin();
            }

        private:
            const test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::const_iterator* _result;
    };


    test_unordered_datastructure::iterator
    non_const_begin_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::iterator* result = createDeviceArray<test_unordered_datastructure::iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         non_const_begin_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    const_begin_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::const_iterator* result = createDeviceArray<test_unordered_datastructure::const_iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         const_begin_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::const_iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::const_iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::const_iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    cbegin_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::const_iterator* result = createDeviceArray<test_unordered_datastructure::const_iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         cbegin_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::const_iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::const_iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::const_iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    begin_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::iterator non_const_iterator   = non_const_begin_iterator(hash_datastructure);
        test_unordered_datastructure::const_iterator const_iterator = const_begin_iterator(hash_datastructure);
        test_unordered_datastructure::const_iterator c_iterator     = cbegin_iterator(hash_datastructure);

        EXPECT_EQ(non_const_iterator, const_iterator);
        EXPECT_EQ(const_iterator, c_iterator);

        return const_iterator;
    }


    class non_const_end_iterator_functor
    {
        public:
            non_const_end_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                           test_unordered_datastructure::iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.end();
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::iterator* _result;
    };


    class const_end_iterator_functor
    {
        public:
            const_end_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                       test_unordered_datastructure::const_iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.end();
            }

        private:
            const test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::const_iterator* _result;
    };


    class cend_iterator_functor
    {
        public:
            cend_iterator_functor(const test_unordered_datastructure& hash_datastructure,
                                  test_unordered_datastructure::const_iterator* result)
                : _hash_datastructure(hash_datastructure),
                  _result(result)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(STDGPU_MAYBE_UNUSED const stdgpu::index_t i)
            {
                *_result = _hash_datastructure.cend();
            }

        private:
            const test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::const_iterator* _result;
    };


    test_unordered_datastructure::iterator
    non_const_end_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::iterator* result = createDeviceArray<test_unordered_datastructure::iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         non_const_end_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    const_end_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::const_iterator* result = createDeviceArray<test_unordered_datastructure::const_iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         const_end_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::const_iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::const_iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::const_iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    cend_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::const_iterator* result = createDeviceArray<test_unordered_datastructure::const_iterator>(1);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(1),
                         cend_iterator_functor(hash_datastructure, result));

        test_unordered_datastructure::const_iterator host_result;
        copyDevice2HostArray<test_unordered_datastructure::const_iterator>(result, 1, &host_result, MemoryCopy::NO_CHECK);

        destroyDeviceArray<test_unordered_datastructure::const_iterator>(result);

        return host_result;
    }


    test_unordered_datastructure::const_iterator
    end_iterator(const test_unordered_datastructure& hash_datastructure)
    {
        test_unordered_datastructure::iterator non_const_iterator   = non_const_end_iterator(hash_datastructure);
        test_unordered_datastructure::const_iterator const_iterator = const_end_iterator(hash_datastructure);
        test_unordered_datastructure::const_iterator c_iterator     = cend_iterator(hash_datastructure);

        EXPECT_EQ(non_const_iterator, const_iterator);
        EXPECT_EQ(const_iterator, c_iterator);

        return const_iterator;
    }


    test_unordered_datastructure::const_iterator
    bucket_iterator(const test_unordered_datastructure& hash_datastructure,
                    const test_unordered_datastructure::key_type& key)
    {
        return begin_iterator(hash_datastructure) + hash_datastructure.bucket(key);
    }
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_no_collision)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2(-5, -15, 13);


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);

    // Found
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_NE(index_2, end_iterator(hash_datastructure));

    // No collisions
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));
    EXPECT_EQ(index_2, bucket_iterator(hash_datastructure, position_2));
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_collision)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2( 7,  3, 15);
    const test_unordered_datastructure::key_type position_3(-5, -15, 13);
    const test_unordered_datastructure::key_type position_4( 5,  15, 13);

    ASSERT_EQ(hash_datastructure.bucket(position_1), hash_datastructure.bucket(position_2));
    ASSERT_EQ(hash_datastructure.bucket(position_3), hash_datastructure.bucket(position_4));


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_3 = insert_key(hash_datastructure, position_3);
    EXPECT_TRUE(inserted_3);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_4 = insert_key(hash_datastructure, position_4);
    EXPECT_TRUE(inserted_4);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);
    test_unordered_datastructure::const_iterator index_3 = find_key(hash_datastructure, position_3);
    test_unordered_datastructure::const_iterator index_4 = find_key(hash_datastructure, position_4);

    // Found
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_NE(index_2, end_iterator(hash_datastructure));
    EXPECT_NE(index_3, end_iterator(hash_datastructure));
    EXPECT_NE(index_4, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_TRUE(contains_key(hash_datastructure, position_2));
    EXPECT_TRUE(contains_key(hash_datastructure, position_3));
    EXPECT_TRUE(contains_key(hash_datastructure, position_4));

    // No collisions
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));
    EXPECT_EQ(index_3, bucket_iterator(hash_datastructure, position_3));

    // Collisions
    EXPECT_NE(index_2, bucket_iterator(hash_datastructure, position_2));
    EXPECT_NE(index_4, bucket_iterator(hash_datastructure, position_4));
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_no_collision)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2(-5, -15, 13);


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);

    // Found
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_NE(index_2, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_TRUE(contains_key(hash_datastructure, position_2));

    // No collisions
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));
    EXPECT_EQ(index_2, bucket_iterator(hash_datastructure, position_2));


    // Erase test data
    bool erased_1 = erase_key(hash_datastructure, position_1);
    EXPECT_TRUE(erased_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool erased_2 = erase_key(hash_datastructure, position_2);
    EXPECT_TRUE(erased_2);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    index_1 = find_key(hash_datastructure, position_1);
    index_2 = find_key(hash_datastructure, position_2);

    // Not found
    EXPECT_EQ(index_1, end_iterator(hash_datastructure));
    EXPECT_EQ(index_2, end_iterator(hash_datastructure));
    EXPECT_FALSE(contains_key(hash_datastructure, position_1));
    EXPECT_FALSE(contains_key(hash_datastructure, position_2));
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_collision)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2( 7,  3, 15);
    const test_unordered_datastructure::key_type position_3(-5, -15, 13);
    const test_unordered_datastructure::key_type position_4( 5,  15, 13);

    ASSERT_EQ(hash_datastructure.bucket(position_1), hash_datastructure.bucket(position_2));
    ASSERT_EQ(hash_datastructure.bucket(position_3), hash_datastructure.bucket(position_4));


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_3 = insert_key(hash_datastructure, position_3);
    EXPECT_TRUE(inserted_3);
    EXPECT_TRUE(hash_datastructure.valid());

    bool inserted_4 = insert_key(hash_datastructure, position_4);
    EXPECT_TRUE(inserted_4);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);
    test_unordered_datastructure::const_iterator index_3 = find_key(hash_datastructure, position_3);
    test_unordered_datastructure::const_iterator index_4 = find_key(hash_datastructure, position_4);

    // Found
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_NE(index_2, end_iterator(hash_datastructure));
    EXPECT_NE(index_3, end_iterator(hash_datastructure));
    EXPECT_NE(index_4, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_TRUE(contains_key(hash_datastructure, position_2));
    EXPECT_TRUE(contains_key(hash_datastructure, position_3));
    EXPECT_TRUE(contains_key(hash_datastructure, position_4));

    // No collisions
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));
    EXPECT_EQ(index_3, bucket_iterator(hash_datastructure, position_3));

    // Collisions
    EXPECT_NE(index_2, bucket_iterator(hash_datastructure, position_2));
    EXPECT_NE(index_4, bucket_iterator(hash_datastructure, position_4));


    // Erase test data
    bool erased_1 = erase_key(hash_datastructure, position_1);
    EXPECT_TRUE(erased_1);
    EXPECT_TRUE(hash_datastructure.valid());

    bool erased_2 = erase_key(hash_datastructure, position_2);
    EXPECT_TRUE(erased_2);
    EXPECT_TRUE(hash_datastructure.valid());

    bool erased_3 = erase_key(hash_datastructure, position_3);
    EXPECT_TRUE(erased_3);
    EXPECT_TRUE(hash_datastructure.valid());

    bool erased_4 = erase_key(hash_datastructure, position_4);
    EXPECT_TRUE(erased_4);
    EXPECT_TRUE(hash_datastructure.valid());

    // Find test data
    index_1 = find_key(hash_datastructure, position_1);
    index_2 = find_key(hash_datastructure, position_2);
    index_3 = find_key(hash_datastructure, position_3);
    index_4 = find_key(hash_datastructure, position_4);

    // Not found
    EXPECT_EQ(index_1, end_iterator(hash_datastructure));
    EXPECT_EQ(index_2, end_iterator(hash_datastructure));
    EXPECT_EQ(index_3, end_iterator(hash_datastructure));
    EXPECT_EQ(index_4, end_iterator(hash_datastructure));
    EXPECT_FALSE(contains_key(hash_datastructure, position_1));
    EXPECT_FALSE(contains_key(hash_datastructure, position_2));
    EXPECT_FALSE(contains_key(hash_datastructure, position_3));
    EXPECT_FALSE(contains_key(hash_datastructure, position_4));
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_double)
{
    const test_unordered_datastructure::key_type position(-7, -3, 15);


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index = find_key(hash_datastructure, position);
    EXPECT_NE(index, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position));
    EXPECT_EQ(index, bucket_iterator(hash_datastructure, position));

    bool inserted_2 = insert_key(hash_datastructure, position);
    EXPECT_FALSE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    index = find_key(hash_datastructure, position);
    EXPECT_NE(index, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position));
    EXPECT_EQ(index, bucket_iterator(hash_datastructure, position));
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_double)
{
    const test_unordered_datastructure::key_type position(-7, -3, 15);


    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index = find_key(hash_datastructure, position);
    EXPECT_NE(index, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position));
    EXPECT_EQ(index, bucket_iterator(hash_datastructure, position));


    // Erase test data
    bool erased_1 = erase_key(hash_datastructure, position);
    EXPECT_TRUE(erased_1);
    EXPECT_TRUE(hash_datastructure.valid());

    index = find_key(hash_datastructure, position);
    EXPECT_EQ(index, end_iterator(hash_datastructure));
    EXPECT_FALSE(contains_key(hash_datastructure, position));

    bool erased_2 = erase_key(hash_datastructure, position);
    EXPECT_FALSE(erased_2);
    EXPECT_TRUE(hash_datastructure.valid());

    index = find_key(hash_datastructure, position);
    EXPECT_EQ(index, end_iterator(hash_datastructure));
    EXPECT_FALSE(contains_key(hash_datastructure, position));
}


namespace
{
    class insert_multiple
    {
        public:
            insert_multiple(const test_unordered_datastructure& hash_datastructure,
                            const test_unordered_datastructure::key_type& key,
                            stdgpu::index_t* inserted)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _inserted(inserted)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                thrust::pair<test_unordered_datastructure::iterator, bool> success = _hash_datastructure.insert(STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_key));

                _inserted[i] = success.second ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            stdgpu::index_t* _inserted;

    };


    class erase_multiple
    {
        public:
            erase_multiple(const test_unordered_datastructure& hash_datastructure,
                           const test_unordered_datastructure::key_type& key,
                           stdgpu::index_t* erased)
                : _hash_datastructure(hash_datastructure),
                  _key(key),
                  _erased(erased)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                bool success = static_cast<bool>(_hash_datastructure.erase(_key));

                _erased[i] = success ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type _key;
            stdgpu::index_t* _erased;
    };


    void
    insert_key_multiple(test_unordered_datastructure& hash_datastructure,
                        const test_unordered_datastructure::key_type& key)
    {
        const stdgpu::index_t old_size = hash_datastructure.size();

        const stdgpu::index_t N = 100000;
        stdgpu::index_t* inserted  = createDeviceArray<stdgpu::index_t>(N);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                         insert_multiple(hash_datastructure, key, inserted));


        stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

        destroyDeviceArray<stdgpu::index_t>(inserted);

        EXPECT_EQ(number_inserted, 1);
        EXPECT_EQ(hash_datastructure.size(), old_size + 1);
        EXPECT_TRUE(hash_datastructure.valid());
    }


    void
    erase_key_multiple(test_unordered_datastructure& hash_datastructure,
                       const test_unordered_datastructure::key_type& key)
    {
        const stdgpu::index_t old_size = hash_datastructure.size();

        const stdgpu::index_t N = 100000;
        stdgpu::index_t* erased = createDeviceArray<stdgpu::index_t>(N);

        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                         erase_multiple(hash_datastructure, key, erased));


        stdgpu::index_t number_erased = thrust::reduce(stdgpu::device_cbegin(erased), stdgpu::device_cend(erased));

        destroyDeviceArray<stdgpu::index_t>(erased);

        EXPECT_EQ(number_erased, 1);
        EXPECT_EQ(hash_datastructure.size(), old_size - 1);
        EXPECT_TRUE(hash_datastructure.valid());
    }
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_multiple_no_collision)
{
    const test_unordered_datastructure::key_type position(-7, -3, 15);

    insert_key_multiple(hash_datastructure, position);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_multiple_no_collision)
{
    const test_unordered_datastructure::key_type position(-7, -3, 15);


    // Insert test data
    bool inserted = insert_key(hash_datastructure, position);
    EXPECT_TRUE(inserted);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index = find_key(hash_datastructure, position);
    EXPECT_NE(index, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position));
    EXPECT_EQ(index, bucket_iterator(hash_datastructure, position));


    erase_key_multiple(hash_datastructure, position);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_multiple_collision)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2( 7,  3, 15);

    ASSERT_EQ(hash_datastructure.bucket(position_1), hash_datastructure.bucket(position_2));

    // Insert test data
    bool inserted = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index = find_key(hash_datastructure, position_1);
    EXPECT_NE(index, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_EQ(index, bucket_iterator(hash_datastructure, position_1));


    insert_key_multiple(hash_datastructure, position_2);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_multiple_collision_head_first)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2( 7,  3, 15);

    ASSERT_EQ(hash_datastructure.bucket(position_1), hash_datastructure.bucket(position_2));

    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);
    EXPECT_NE(index_2, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_2));
    EXPECT_NE(index_2, bucket_iterator(hash_datastructure, position_2));


    erase_key_multiple(hash_datastructure, position_1);
    erase_key_multiple(hash_datastructure, position_2);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_multiple_collision_tail_first)
{
    const test_unordered_datastructure::key_type position_1(-7, -3, 15);
    const test_unordered_datastructure::key_type position_2( 7,  3, 15);

    ASSERT_EQ(hash_datastructure.bucket(position_1), hash_datastructure.bucket(position_2));

    // Insert test data
    bool inserted_1 = insert_key(hash_datastructure, position_1);
    EXPECT_TRUE(inserted_1);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index_1 = find_key(hash_datastructure, position_1);
    EXPECT_NE(index_1, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_1));
    EXPECT_EQ(index_1, bucket_iterator(hash_datastructure, position_1));

    bool inserted_2 = insert_key(hash_datastructure, position_2);
    EXPECT_TRUE(inserted_2);
    EXPECT_TRUE(hash_datastructure.valid());

    test_unordered_datastructure::const_iterator index_2 = find_key(hash_datastructure, position_2);
    EXPECT_NE(index_2, end_iterator(hash_datastructure));
    EXPECT_TRUE(contains_key(hash_datastructure, position_2));
    EXPECT_NE(index_2, bucket_iterator(hash_datastructure, position_2));


    erase_key_multiple(hash_datastructure, position_2);
    erase_key_multiple(hash_datastructure, position_1);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_while_full)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(1);

    // Fill tiny hash table
    const test_unordered_datastructure::key_type position_1(1, 2, 3);
    const test_unordered_datastructure::key_type position_2(4, 5, 6);

    insert_key(tiny_hash_datastructure, position_1);
    insert_key(tiny_hash_datastructure, position_2);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    // Insert entry in full hash table
    const test_unordered_datastructure::key_type position_3(7, 8, 9);

    bool inserted_3 = insert_key(tiny_hash_datastructure, position_3);
    EXPECT_FALSE(inserted_3);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_multiple_while_full)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(1);

    // Fill tiny hash table
    const test_unordered_datastructure::key_type position_1(1, 2, 3);
    const test_unordered_datastructure::key_type position_2(4, 5, 6);

    insert_key(tiny_hash_datastructure, position_1);
    insert_key(tiny_hash_datastructure, position_2);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    // Multi-insert entry in full hash table
    const test_unordered_datastructure::key_type position_3(7, 8, 9);


    const stdgpu::index_t N = 100000;
    stdgpu::index_t* inserted  = createDeviceArray<stdgpu::index_t>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     insert_multiple(tiny_hash_datastructure, position_3, inserted));


    stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

    destroyDeviceArray<stdgpu::index_t>(inserted);

    EXPECT_EQ(number_inserted, 0);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_while_excess_empty)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(2);

    // Fill tiny hash table
    const test_unordered_datastructure::key_type position_1( 1,  2,  3);
    const test_unordered_datastructure::key_type position_2(-1,  2,  3);
    const test_unordered_datastructure::key_type position_3( 1, -2,  3);

    insert_key(tiny_hash_datastructure, position_1);
    insert_key(tiny_hash_datastructure, position_2);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);


    bool inserted_3 = insert_key(tiny_hash_datastructure, position_3);
    EXPECT_FALSE(inserted_3);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


namespace
{
    class insert_keys
    {
        public:
            insert_keys(const test_unordered_datastructure& hash_datastructure,
                        test_unordered_datastructure::key_type* keys,
                        stdgpu::index_t* inserted)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _inserted(inserted)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                thrust::pair<test_unordered_datastructure::iterator, bool> success = _hash_datastructure.insert(STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_keys[i]));

                _inserted[i] = success.second ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            stdgpu::index_t* _inserted;
    };


    class emplace_keys
    {
        public:
            emplace_keys(const test_unordered_datastructure& hash_datastructure,
                         test_unordered_datastructure::key_type* keys,
                         stdgpu::index_t* inserted)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _inserted(inserted)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                thrust::pair<test_unordered_datastructure::iterator, bool> success = _hash_datastructure.emplace(STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_keys[i]));

                _inserted[i] = success.second ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            stdgpu::index_t* _inserted;
    };


    class erase_keys
    {
        public:
            erase_keys(const test_unordered_datastructure& hash_datastructure,
                       test_unordered_datastructure::key_type* keys,
                       stdgpu::index_t* erased)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _erased(erased)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                bool success = static_cast<bool>(_hash_datastructure.erase(_keys[i]));

                _erased[i] = success ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            stdgpu::index_t* _erased;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_parallel_while_one_free)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(1);

    // Fill tiny hash table and only keep one free
    const test_unordered_datastructure::key_type position_1(1, 2, 3);

    insert_key(tiny_hash_datastructure, position_1);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_FALSE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 1);


    const stdgpu::index_t N = 100000;

    // Generate true random numbers
    size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<std::int16_t> dist(std::numeric_limits<std::int16_t>::lowest(), std::numeric_limits<std::int16_t>::max());

    test_unordered_datastructure::key_type* host_positions = createHostArray<test_unordered_datastructure::key_type>(N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        test_unordered_datastructure::key_type random(dist(rng), dist(rng), dist(rng));

        host_positions[i] = random;
    }


    // Multi-insert entry in full hash table
    stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
    test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     insert_keys(tiny_hash_datastructure, positions, inserted));


    stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

    EXPECT_EQ(number_inserted, 1);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);


    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyDeviceArray<stdgpu::index_t>(inserted);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_parallel_while_excess_empty)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(2);

    // Fill tiny hash table
    const test_unordered_datastructure::key_type position_1( 1,  2,  3);
    const test_unordered_datastructure::key_type position_2(-1,  2,  3);
    const test_unordered_datastructure::key_type position_3( 1, -2,  3);

    insert_key(tiny_hash_datastructure, position_1);
    insert_key(tiny_hash_datastructure, position_2);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);


    const stdgpu::index_t N = 100000;

    // Multi-insert entry in full hash table
    stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
    test_unordered_datastructure::key_type* positions   = createDeviceArray<test_unordered_datastructure::key_type>(N, position_3);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     insert_keys(tiny_hash_datastructure, positions, inserted));


    stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));


    EXPECT_EQ(number_inserted, 0);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyDeviceArray<stdgpu::index_t>(inserted);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, emplace_parallel_while_one_free)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(1);

    // Fill tiny hash table and only keep one free
    const test_unordered_datastructure::key_type position_1(1, 2, 3);

    insert_key(tiny_hash_datastructure, position_1);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_FALSE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 1);


    const stdgpu::index_t N = 100000;

    // Generate true random numbers
    size_t seed = test_utils::random_seed();

    std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
    std::uniform_int_distribution<std::int16_t> dist(std::numeric_limits<std::int16_t>::lowest(), std::numeric_limits<std::int16_t>::max());

    test_unordered_datastructure::key_type* host_positions = createHostArray<test_unordered_datastructure::key_type>(N);

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        test_unordered_datastructure::key_type random(dist(rng), dist(rng), dist(rng));

        host_positions[i] = random;
    }


    // Multi-insert entry in full hash table
    stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
    test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_keys(tiny_hash_datastructure, positions, inserted));


    stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

    EXPECT_EQ(number_inserted, 1);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_TRUE(tiny_hash_datastructure.full());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);


    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyDeviceArray<stdgpu::index_t>(inserted);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, emplace_parallel_while_excess_empty)
{
    test_unordered_datastructure tiny_hash_datastructure = test_unordered_datastructure::createDeviceObject(2);

    // Fill tiny hash table
    const test_unordered_datastructure::key_type position_1( 1,  2,  3);
    const test_unordered_datastructure::key_type position_2(-1,  2,  3);
    const test_unordered_datastructure::key_type position_3( 1, -2,  3);

    insert_key(tiny_hash_datastructure, position_1);
    insert_key(tiny_hash_datastructure, position_2);

    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);


    const stdgpu::index_t N = 100000;

    // Multi-insert entry in full hash table
    stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
    test_unordered_datastructure::key_type* positions   = createDeviceArray<test_unordered_datastructure::key_type>(N, position_3);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     emplace_keys(tiny_hash_datastructure, positions, inserted));


    stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));


    EXPECT_EQ(number_inserted, 0);
    EXPECT_TRUE(tiny_hash_datastructure.valid());
    EXPECT_EQ(tiny_hash_datastructure.size(), 2);

    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyDeviceArray<stdgpu::index_t>(inserted);

    test_unordered_datastructure::destroyDeviceObject(tiny_hash_datastructure);
}


namespace
{
    test_unordered_datastructure::key_type*
    create_unique_random_host_keys(const stdgpu::index_t N)
    {
        // Generate true random numbers
        size_t seed = test_utils::random_seed();

        std::default_random_engine rng(static_cast<std::default_random_engine::result_type>(seed));
        std::uniform_int_distribution<std::int16_t> dist(std::numeric_limits<std::int16_t>::lowest(), std::numeric_limits<std::int16_t>::max());

        test_unordered_datastructure::key_type* host_positions = createHostArray<test_unordered_datastructure::key_type>(N);

        std::unordered_set<test_unordered_datastructure::key_type, test_unordered_datastructure::hasher> set;
        set.reserve(static_cast<std::size_t>(N));
        while (static_cast<stdgpu::index_t>(set.size()) < N)
        {
            test_unordered_datastructure::key_type random(dist(rng), dist(rng), dist(rng));

            if (set.insert(random).second)
            {
                host_positions[set.size() - 1] = random;
            }
        }

        return host_positions;
    }


    test_unordered_datastructure::key_type*
    insert_unique_parallel(test_unordered_datastructure& hash_datastructure,
                           const stdgpu::index_t N)
    {
        test_unordered_datastructure::key_type* host_positions = create_unique_random_host_keys(N);

        stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
        test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

        thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                         insert_keys(hash_datastructure, positions, inserted));


        stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

        EXPECT_EQ(number_inserted, N);
        EXPECT_FALSE(hash_datastructure.empty());
        EXPECT_EQ(hash_datastructure.size(), N);
        EXPECT_TRUE(hash_datastructure.valid());


        destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
        destroyDeviceArray<stdgpu::index_t>(inserted);

        return host_positions;
    }


    test_unordered_datastructure::key_type*
    emplace_unique_parallel(test_unordered_datastructure& hash_datastructure,
                            const stdgpu::index_t N)
    {
        test_unordered_datastructure::key_type* host_positions = create_unique_random_host_keys(N);

        stdgpu::index_t* inserted                           = createDeviceArray<stdgpu::index_t>(N);
        test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

        thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                         emplace_keys(hash_datastructure, positions, inserted));


        stdgpu::index_t number_inserted = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));

        EXPECT_EQ(number_inserted, N);
        EXPECT_FALSE(hash_datastructure.empty());
        EXPECT_EQ(hash_datastructure.size(), N);
        EXPECT_TRUE(hash_datastructure.valid());


        destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
        destroyDeviceArray<stdgpu::index_t>(inserted);

        return host_positions;
    }


    void
    erase_unique_parallel(test_unordered_datastructure& hash_datastructure,
                          test_unordered_datastructure::key_type* host_positions,
                          const stdgpu::index_t N)
    {
        stdgpu::index_t* erased                             = createDeviceArray<stdgpu::index_t>(N);
        test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

        thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(N),
                         erase_keys(hash_datastructure, positions, erased));


        stdgpu::index_t number_erased = thrust::reduce(stdgpu::device_cbegin(erased), stdgpu::device_cend(erased));

        EXPECT_EQ(number_erased, N);
        EXPECT_TRUE(hash_datastructure.empty());
        EXPECT_EQ(hash_datastructure.size(), 0);


        destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
        destroyDeviceArray<stdgpu::index_t>(erased);
    }

    class Key2ValueFunctor
    {
        public:
            Key2ValueFunctor(const test_unordered_datastructure& hash_datastructure,
                             test_unordered_datastructure::key_type* keys,
                             test_unordered_datastructure::value_type* values)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _values(values)
            {

            }

            STDGPU_HOST_DEVICE void
            operator()(const stdgpu::index_t i)
            {
                test_unordered_datastructure::allocator_type a = _hash_datastructure.get_allocator();
                stdgpu::allocator_traits<test_unordered_datastructure::allocator_type>::construct(a,
                                                                                                  &(_values[i]),
                                                                                                  STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_keys[i]));
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            test_unordered_datastructure::value_type* _values;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);

    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, emplace_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = emplace_unique_parallel(hash_datastructure, N);

    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);

    erase_unique_parallel(hash_datastructure, host_positions, N);

    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_range_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions  = create_unique_random_host_keys(N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);
    test_unordered_datastructure::value_type* values        = createDeviceArray<test_unordered_datastructure::value_type>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     Key2ValueFunctor(hash_datastructure, positions, values));

    stdgpu::device_ptr<test_unordered_datastructure::value_type> values_begin   = stdgpu::device_begin(values);
    stdgpu::device_ptr<test_unordered_datastructure::value_type> values_end     = stdgpu::device_end(values);
    hash_datastructure.insert(values_begin, values_end);

    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyDeviceArray<test_unordered_datastructure::value_type>(values);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_const_range_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions  = create_unique_random_host_keys(N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);
    test_unordered_datastructure::value_type* values        = createDeviceArray<test_unordered_datastructure::value_type>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     Key2ValueFunctor(hash_datastructure, positions, values));

    stdgpu::device_ptr<const test_unordered_datastructure::value_type> values_begin = stdgpu::device_cbegin(values);
    stdgpu::device_ptr<const test_unordered_datastructure::value_type> values_end   = stdgpu::device_cend(values);
    hash_datastructure.insert(values_begin, values_end);

    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyDeviceArray<test_unordered_datastructure::value_type>(values);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_range_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions  = create_unique_random_host_keys(N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);
    test_unordered_datastructure::value_type* values        = createDeviceArray<test_unordered_datastructure::value_type>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     Key2ValueFunctor(hash_datastructure, positions, values));

    stdgpu::device_ptr<test_unordered_datastructure::value_type> values_begin   = stdgpu::device_begin(values);
    stdgpu::device_ptr<test_unordered_datastructure::value_type> values_end     = stdgpu::device_end(values);
    hash_datastructure.insert(values_begin, values_end);

    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());

    stdgpu::device_ptr<test_unordered_datastructure::key_type> positions_begin  = stdgpu::device_begin(positions);
    stdgpu::device_ptr<test_unordered_datastructure::key_type> positions_end    = stdgpu::device_end(positions);
    hash_datastructure.erase(positions_begin, positions_end);

    EXPECT_TRUE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), 0);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyDeviceArray<test_unordered_datastructure::value_type>(values);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, erase_const_range_unique_parallel)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions  = create_unique_random_host_keys(N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);
    test_unordered_datastructure::value_type* values        = createDeviceArray<test_unordered_datastructure::value_type>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     Key2ValueFunctor(hash_datastructure, positions, values));

    stdgpu::device_ptr<const test_unordered_datastructure::value_type> values_begin = stdgpu::device_cbegin(values);
    stdgpu::device_ptr<const test_unordered_datastructure::value_type> values_end   = stdgpu::device_cend(values);
    hash_datastructure.insert(values_begin, values_end);

    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());

    stdgpu::device_ptr<const test_unordered_datastructure::key_type> positions_begin    = stdgpu::device_cbegin(positions);
    stdgpu::device_ptr<const test_unordered_datastructure::key_type> positions_end      = stdgpu::device_cend(positions);
    hash_datastructure.erase(positions_begin, positions_end);

    EXPECT_TRUE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), 0);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyDeviceArray<test_unordered_datastructure::value_type>(values);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


namespace
{
    class insert_and_erase_keys
    {
        public:
            insert_and_erase_keys(const test_unordered_datastructure& hash_datastructure,
                                  test_unordered_datastructure::key_type* keys,
                                  stdgpu::index_t* inserted,
                                  stdgpu::index_t* erased)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _inserted(inserted),
                  _erased(erased)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                thrust::pair<test_unordered_datastructure::iterator, bool> success_insert = _hash_datastructure.insert(STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE(_keys[i]));

                _inserted[i] = success_insert.second ? 1 : 0;

                bool success_erase = static_cast<bool>(_hash_datastructure.erase(_keys[i]));

                _erased[i] = success_erase ? 1 : 0;
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            stdgpu::index_t* _inserted;
            stdgpu::index_t* _erased;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, insert_and_erase_unique_parallel)
{
    const stdgpu::index_t N = 10000;

    test_unordered_datastructure::key_type* host_positions = create_unique_random_host_keys(N);

    stdgpu::index_t* inserted   = createDeviceArray<stdgpu::index_t>(N);
    stdgpu::index_t* erased     = createDeviceArray<stdgpu::index_t>(N);
    test_unordered_datastructure::key_type* positions   = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     insert_and_erase_keys(hash_datastructure, positions, inserted, erased));


    stdgpu::index_t number_inserted    = thrust::reduce(stdgpu::device_cbegin(inserted), stdgpu::device_cend(inserted));
    stdgpu::index_t number_erased      = thrust::reduce(stdgpu::device_cbegin(erased),   stdgpu::device_cend(erased));

    EXPECT_EQ(number_inserted, N);
    EXPECT_EQ(number_erased, N);
    EXPECT_TRUE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), 0);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyDeviceArray<stdgpu::index_t>(erased);
    destroyDeviceArray<stdgpu::index_t>(inserted);

    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


namespace
{
    class store_bucket_sizes
    {
        public:
            store_bucket_sizes(const test_unordered_datastructure& hash_datastructure,
                               stdgpu::index_t* bucket_sizes)
                : _hash_datastructure(hash_datastructure),
                  _bucket_sizes(bucket_sizes)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                _bucket_sizes[i] = _hash_datastructure.bucket_size(i);
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            stdgpu::index_t* _bucket_sizes;
    };


    class store_counts
    {
        public:
            store_counts(const test_unordered_datastructure& hash_datastructure,
                         test_unordered_datastructure::key_type* keys,
                         stdgpu::index_t* counts)
                : _hash_datastructure(hash_datastructure),
                  _keys(keys),
                  _counts(counts)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const stdgpu::index_t i)
            {
                _counts[i] = _hash_datastructure.count(_keys[i]);
            }

        private:
            test_unordered_datastructure _hash_datastructure;
            test_unordered_datastructure::key_type* _keys;
            stdgpu::index_t* _counts;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, bucket_size_sum)
{
    const stdgpu::index_t N = 10000;

    test_unordered_datastructure::key_type* host_positions  = insert_unique_parallel(hash_datastructure, N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

    stdgpu::index_t* bucket_sizes = createDeviceArray<stdgpu::index_t>(hash_datastructure.bucket_count());

    thrust::for_each(thrust::counting_iterator<stdgpu::index_t>(0), thrust::counting_iterator<stdgpu::index_t>(hash_datastructure.bucket_count()),
                     store_bucket_sizes(hash_datastructure, bucket_sizes));

    stdgpu::index_t bucket_size_sum = thrust::reduce(stdgpu::device_cbegin(bucket_sizes), stdgpu::device_cend(bucket_sizes));

    EXPECT_EQ(bucket_size_sum, N);
    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());

    destroyDeviceArray<stdgpu::index_t>(bucket_sizes);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, count_sum)
{
    const stdgpu::index_t N = 10000;

    test_unordered_datastructure::key_type* host_positions  = insert_unique_parallel(hash_datastructure, N);
    test_unordered_datastructure::key_type* positions       = copyCreateHost2DeviceArray<test_unordered_datastructure::key_type>(host_positions, N);

    stdgpu::index_t* counts = createDeviceArray<stdgpu::index_t>(N);

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N),
                     store_counts(hash_datastructure, positions, counts));

    stdgpu::index_t counts_sum = thrust::reduce(stdgpu::device_cbegin(counts), stdgpu::device_cend(counts));

    EXPECT_EQ(counts_sum, N);
    EXPECT_FALSE(hash_datastructure.empty());
    EXPECT_EQ(hash_datastructure.size(), N);
    EXPECT_TRUE(hash_datastructure.valid());

    destroyDeviceArray<stdgpu::index_t>(counts);
    destroyDeviceArray<test_unordered_datastructure::key_type>(positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


namespace
{
    class for_each_counter
    {
        public:
            for_each_counter(const stdgpu::atomic<unsigned int>& counter,
                             const stdgpu::atomic<unsigned int>& bad_counter,
                             const test_unordered_datastructure& hash_datastructure)
                : _counter(counter),
                  _bad_counter(bad_counter),
                  _hash_datastructure(hash_datastructure)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const test_unordered_datastructure::value_type& value)
            {
                if (!_hash_datastructure.contains(STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY(value)))
                {
                    ++_bad_counter;
                }

                ++_counter;
            }

        private:
            stdgpu::atomic<unsigned int> _counter;
            stdgpu::atomic<unsigned int> _bad_counter;
            test_unordered_datastructure _hash_datastructure;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, range_for_each_count)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);

    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);


    stdgpu::atomic<unsigned int> counter     = stdgpu::atomic<unsigned int>::createDeviceObject();
    stdgpu::atomic<unsigned int> bad_counter = stdgpu::atomic<unsigned int>::createDeviceObject();

    auto range = hash_datastructure.device_range();
    thrust::for_each(range.begin(), range.end(),
                     for_each_counter(counter, bad_counter, hash_datastructure));

    EXPECT_EQ(counter.load(), static_cast<unsigned int>(hash_datastructure.size()));
    EXPECT_EQ(bad_counter.load(), static_cast<unsigned int>(0));

    stdgpu::atomic<unsigned int>::destroyDeviceObject(counter);
    stdgpu::atomic<unsigned int>::destroyDeviceObject(bad_counter);
}


namespace
{
    class insert_vector
    {
        public:
            explicit insert_vector(const stdgpu::vector<test_unordered_datastructure::key_type>& keys)
                : _keys(keys)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const test_unordered_datastructure::value_type& value)
            {
                _keys.push_back(STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY(value));
            }

        private:
            stdgpu::vector<test_unordered_datastructure::key_type> _keys;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, range_for_each_keys_same)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);


    stdgpu::vector<test_unordered_datastructure::key_type> keys = stdgpu::vector<test_unordered_datastructure::key_type>::createDeviceObject(N);

    auto range = hash_datastructure.device_range();
    thrust::for_each(range.begin(), range.end(),
                     insert_vector(keys));

    ASSERT_EQ(keys.size(), N);

    test_unordered_datastructure::key_type* host_positions_inserted = copyCreateDevice2HostArray<test_unordered_datastructure::key_type>(keys.data(), keys.size());

    thrust::sort(host_positions,          host_positions + N,          less());
    thrust::sort(host_positions_inserted, host_positions_inserted + N, less());

    for (stdgpu::index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host_positions[i], host_positions_inserted[i]);
    }


    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
    destroyHostArray<test_unordered_datastructure::key_type>(host_positions_inserted);
    stdgpu::vector<test_unordered_datastructure::key_type>::destroyDeviceObject(keys);
}


namespace
{
    class erase_hash
    {
        public:
            explicit erase_hash(const test_unordered_datastructure& hash_datastructure)
                : _hash_datastructure(hash_datastructure)
            {

            }

            STDGPU_DEVICE_ONLY void
            operator()(const test_unordered_datastructure::value_type& value)
            {
                _hash_datastructure.erase(STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY(value));
            }

        private:
            test_unordered_datastructure _hash_datastructure;
    };
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, range_for_each_erase)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);


    auto range = hash_datastructure.device_range();
    thrust::for_each(range.begin(), range.end(),
                     erase_hash(hash_datastructure));


    EXPECT_EQ(hash_datastructure.size(), 0);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, clear)
{
    const stdgpu::index_t N = 100000;

    test_unordered_datastructure::key_type* host_positions = insert_unique_parallel(hash_datastructure, N);


    hash_datastructure.clear();


    EXPECT_EQ(hash_datastructure.size(), 0);
    EXPECT_TRUE(hash_datastructure.valid());


    destroyHostArray<test_unordered_datastructure::key_type>(host_positions);
}


TEST_F(STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS, deprecated_createDeviceObject)
{
    const stdgpu::index_t buckets = static_cast<stdgpu::index_t>(pow(2, 17));
    const stdgpu::index_t excess  = 1000;

    test_unordered_datastructure deprecated_hash_datastructure = test_unordered_datastructure::createDeviceObject(buckets, excess);

    EXPECT_EQ(deprecated_hash_datastructure.bucket_count(), buckets);
    EXPECT_EQ(deprecated_hash_datastructure.excess_count(), excess);
    EXPECT_EQ(deprecated_hash_datastructure.total_count(),  buckets + excess);

    test_unordered_datastructure::destroyDeviceObject(deprecated_hash_datastructure);
}


