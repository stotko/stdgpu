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

#include <stdgpu/unordered_set.cuh>

#include <cstddef>

#include <stdgpu/platform.h>
#include <test_memory_utils.h>

struct int_hash
{
    using is_transparent = void;

    template <typename T>
    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const T& key) const
    {
        return stdgpu::hash<T>{}(key);
    }
};

// Explicit template instantiations
namespace stdgpu
{

template class unordered_set<int>;

// Instantiation of variadic templates emit warnings in CUDA backend
/*
template
STDGPU_DEVICE_ONLY pair<typename unordered_set<int>::iterator, bool>
unordered_set<int>::emplace<int>(int&&);
*/

template void unordered_set<int>::insert(device_ptr<const typename unordered_set<int>::value_type>,
                                         device_ptr<const typename unordered_set<int>::value_type>);

template void unordered_set<int>::erase(device_ptr<const typename unordered_set<int>::key_type>,
                                        device_ptr<const typename unordered_set<int>::key_type>);

template STDGPU_DEVICE_ONLY typename unordered_set<int, int_hash, equal_to<>>::index_type
unordered_set<int, int_hash, equal_to<>>::count<short>(const short&) const;

template STDGPU_DEVICE_ONLY typename unordered_set<int, int_hash, equal_to<>>::iterator
unordered_set<int, int_hash, equal_to<>>::find<short>(const short&);

template STDGPU_DEVICE_ONLY typename unordered_set<int, int_hash, equal_to<>>::const_iterator
unordered_set<int, int_hash, equal_to<>>::find<short>(const short&) const;

template STDGPU_DEVICE_ONLY bool
unordered_set<int, int_hash, equal_to<>>::contains<short>(const short&) const;

} // namespace stdgpu

struct vec3int16
{
    vec3int16() = default;

    STDGPU_HOST_DEVICE
    vec3int16(const std::int16_t new_x, const std::int16_t new_y, const std::int16_t new_z)
      : x(new_x)
      , y(new_y)
      , z(new_z)
    {
    }

    std::int16_t x = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::int16_t y = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::int16_t z = 0; // NOLINT(misc-non-private-member-variables-in-classes)
};

inline STDGPU_HOST_DEVICE bool
operator==(const vec3int16& lhs, const vec3int16& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

struct less
{
    inline STDGPU_HOST_DEVICE bool
    operator()(const vec3int16& lhs, const vec3int16& rhs) const
    {
        if (lhs.x < rhs.x)
        {
            return true;
        }
        if (lhs.x > rhs.x)
        {
            return false;
        }

        if (lhs.y < rhs.y)
        {
            return true;
        }
        if (lhs.y > rhs.y)
        {
            return false;
        }

        if (lhs.z < rhs.z)
        {
            return true;
        }
        if (lhs.z > rhs.z)
        {
            return false;
        }

        return true;
    }
};

struct vec3int32
{
    vec3int32() = default;

    STDGPU_HOST_DEVICE
    vec3int32(const std::int32_t new_x, const std::int32_t new_y, const std::int32_t new_z)
      : x(new_x)
      , y(new_y)
      , z(new_z)
    {
    }

    STDGPU_HOST_DEVICE
    ~vec3int32() // NOLINT(hicpp-use-equals-default,modernize-use-equals-default)
    {
        // nontrivial destructor
    }

    vec3int32(const vec3int32&) = default;
    vec3int32&
    operator=(const vec3int32&) = default;

    vec3int32(vec3int32&&) = default;
    vec3int32&
    operator=(vec3int32&&) = default;

    std::int32_t x = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::int32_t y = 0; // NOLINT(misc-non-private-member-variables-in-classes)
    std::int32_t z = 0; // NOLINT(misc-non-private-member-variables-in-classes)
};

inline STDGPU_HOST_DEVICE bool
operator==(const vec3int32& lhs, const vec3int32& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline STDGPU_HOST_DEVICE bool
operator==(const vec3int16& lhs, const vec3int32& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline STDGPU_HOST_DEVICE bool
operator==(const vec3int32& lhs, const vec3int16& rhs)
{
    return rhs == lhs;
}

struct vec_hash
{
    using is_transparent = void;

    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const vec3int16& key) const
    {
        const std::size_t prime_x = static_cast<std::size_t>(73856093U);
        const std::size_t prime_y = static_cast<std::size_t>(19349669U);
        const std::size_t prime_z = static_cast<std::size_t>(83492791U);

        return (static_cast<std::size_t>(key.x) * prime_x) ^ (static_cast<std::size_t>(key.y) * prime_y) ^
               (static_cast<std::size_t>(key.z) * prime_z);
    }

    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const vec3int32& key) const
    {
        const std::size_t prime_x = static_cast<std::size_t>(73856093U);
        const std::size_t prime_y = static_cast<std::size_t>(19349669U);
        const std::size_t prime_z = static_cast<std::size_t>(83492791U);

        return (static_cast<std::size_t>(key.x) * prime_x) ^ (static_cast<std::size_t>(key.y) * prime_y) ^
               (static_cast<std::size_t>(key.z) * prime_z);
    }
};

inline STDGPU_HOST_DEVICE vec3int16
key_to_value(const vec3int16& key)
{
    return key;
}

inline STDGPU_HOST_DEVICE vec3int32
key_to_value(const vec3int32& key)
{
    return key;
}

inline STDGPU_HOST_DEVICE vec3int16
value_to_key(const vec3int16& key)
{
    return key;
}

inline STDGPU_HOST_DEVICE vec3int32
key_to_keylike(const vec3int16& key)
{
    return { key.x, key.y, key.z };
}

#define STDGPU_UNORDERED_DATASTRUCTURE_TEST_CLASS stdgpu_unordered_set
#define STDGPU_UNORDERED_DATASTRUCTURE_TYPE stdgpu::unordered_set<vec3int16, vec_hash, stdgpu::equal_to<>>
#define STDGPU_UNORDERED_DATASTRUCTURE_KEY2VALUE key_to_value
#define STDGPU_UNORDERED_DATASTRUCTURE_VALUE2KEY value_to_key
#define STDGPU_UNORDERED_DATASTRUCTURE_TRANSPARENT_KEYTYPE vec3int32
#define STDGPU_UNORDERED_DATASTRUCTURE_KEY2KEYLIKE key_to_keylike
#define STDGPU_UNORDERED_DATASTRUCTURE_NONTRIVIAL_TYPE stdgpu::unordered_set<vec3int32, vec_hash, stdgpu::equal_to<>>
#define STDGPU_UNORDERED_DATASTRUCTURE_CUSTOM_ALLOCATOR test_utils::test_device_allocator<vec3int16>
#define STDGPU_UNORDERED_DATASTRUCTURE_CUSTOM_TYPE                                                                     \
    stdgpu::unordered_set<vec3int16, vec_hash, stdgpu::equal_to<>, STDGPU_UNORDERED_DATASTRUCTURE_CUSTOM_ALLOCATOR>

#include "unordered_datastructure.inc"
