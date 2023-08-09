/*
 *  Copyright 2021 Patrick Stotko
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

#ifndef TEST_MEMORY_UTILS_H
#define TEST_MEMORY_UTILS_H

#include <stdgpu/cstddef.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>

namespace test_utils
{
/**
 * \brief A statistics class for allocators
 */
struct allocator_statistics
{
    /**
     * \brief Resets the statistics
     */
    void
    reset();

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    stdgpu::index_t default_constructions = 0; /**< The number of default constructions */
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    stdgpu::index_t copy_constructions = 0; /**< The number of copy constructions */
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    stdgpu::index_t destructions = 0; /**< The number of destructions */
};

/**
 * \brief Returns the global allocator statistics
 */
allocator_statistics&
get_allocator_statistics();

/**
 * \brief A test allocator for device memory
 * \tparam T A type
 */
template <typename T>
class test_device_allocator
{
public:
    using base_type = stdgpu::safe_device_allocator<T>; /**< stdgpu::safe_device_allocator<T> */
    using value_type = typename base_type::value_type;  /**< base_type::value_type */

    /**
     * \brief Default constructor
     */
    STDGPU_HOST_DEVICE
    test_device_allocator() noexcept;

    /**
     * \brief Destructor
     */
    STDGPU_HOST_DEVICE
    ~test_device_allocator() noexcept;

    /**
     * \brief Copy constructor
     * \param[in] other The allocator to be copied from
     */
    STDGPU_HOST_DEVICE
    test_device_allocator(const test_device_allocator& other) noexcept;

    /**
     * \brief Deleted copy assignment operator
     */
    test_device_allocator&
    operator=(const test_device_allocator&) = delete;

    /**
     * \brief Copy constructor
     * \tparam U Another type
     * \param[in] other The allocator to be copied from
     */
    template <typename U>
    explicit STDGPU_HOST_DEVICE
    test_device_allocator(const test_device_allocator<U>& other) noexcept;

    /**
     * \brief Deleted move constructor
     */
    test_device_allocator(test_device_allocator&&) = delete;

    /**
     * \brief Deleted move assignment operator
     */
    test_device_allocator&
    operator=(test_device_allocator&&) = delete;

    /**
     * \brief Allocates a memory block of the given size
     * \param[in] n The size of the memory block in bytes
     * \return A pointer to the allocated memory block
     */
    [[nodiscard]] T*
    allocate(stdgpu::index64_t n);

    /**
     * \brief Deallocates the given memory block
     * \param[in] p A pointer to the memory block
     * \param[in] n The size of the memory block in bytes (must match the size during allocation)
     */
    void
    deallocate(T* p, stdgpu::index64_t n);
};
} // namespace test_utils

#include <test_memory_utils_detail.h>

#endif // TEST_MEMORY_UTILS_H
