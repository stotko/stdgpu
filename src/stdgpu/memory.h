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

#ifndef STDGPU_MEMORY_H
#define STDGPU_MEMORY_H

/**
 * \file stdgpu/memory.h
 */

#include <stdgpu/attribute.h>
#include <stdgpu/config.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>



/**
 * \brief The place to initialize the created array
 */
enum class Initialization
{
    HOST,           /**< The array is initialized on the host (CPU) */
    DEVICE          /**< The array is initialized on the device (GPU) */
};


/**
 * \brief Creates a new device array and initializes (fills) it with the given default value
 * \tparam T The type of the array
 * \param[in] count The number of elements of the new array
 * \param[in] default_value A default value, that should be stored in every array entry
 * \return The allocated device array if count > 0, nullptr otherwise
 * \post get_dynamic_memory_type(result) == dynamic_memory_type::device if count > 0
 * \note If `STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING` is defined, this functions prints a warning when the array initialization requires using an auxiliary host array (i.e. to support compilation without a device compiler as a .cpp file).
 */
template <typename T>
T*
createDeviceArray(const stdgpu::index64_t count,
                  const T default_value = T());


/**
 * \brief Creates a new host array and initializes (fills) it with the given default value
 * \tparam T The type of the array
 * \param[in] count The number of elements of the new array
 * \param[in] default_value A default value, that should be stored in every array entry
 * \return The allocated host array if count > 0, nullptr otherwise
 * \post get_dynamic_memory_type(result) == dynamic_memory_type::device if count > 0
 */
template <typename T>
T*
createHostArray(const stdgpu::index64_t count,
                const T default_value = T());


/**
 * \brief Creates a new managed array and initializes (fills) it with the given default value
 * \tparam T The type of the array
 * \param[in] count The number of elements of the new array
 * \param[in] default_value A default value, that should be stored in every array entry
 * \param[in] initialize_on The device on which the fill operation is performed
 * \return The allocated managed array if count > 0, nullptr otherwise
 * \post get_dynamic_memory_type(result) == dynamic_memory_type::managed if count > 0
 * \note If `STDGPU_ENABLE_MANAGED_ARRAY_WARNING` is defined, this functions prints a warning when device initialization is not possible and initialization on the host is performed instead (i.e. to support compilation without a device compiler as a .cpp file).
 */
template <typename T>
T*
createManagedArray(const stdgpu::index64_t count,
                   const T default_value = T(),
                   const Initialization initialize_on = Initialization::DEVICE);


/**
 * \brief Destroys the given device array
 * \tparam T The type of the array
 * \param[in] device_array A device array
 */
template <typename T>
void
destroyDeviceArray(T*& device_array);


/**
 * \brief Destroys the given host array
 * \tparam T The type of the array
 * \param[in] host_array A host array
 */
template <typename T>
void
destroyHostArray(T*& host_array);


/**
 * \brief Destroys the given managed array
 * \tparam T The type of the array
 * \param[in] managed_array A managed array
 */
template <typename T>
void
destroyManagedArray(T*& managed_array);



/**
 * \brief The copy check states
 */
enum class MemoryCopy
{
    NO_CHECK,       /**< No checks should be performed. This is useful when copying from/to arrays not created by our API, e.g. created by 3rd party libraries or pointers to local variables. */
    RANGE_CHECK     /**< The range of the source array is checked to fit inside the range of the target array. */
};


/**
 * \brief Creates and copies the given device array to the host
 * \tparam T The type of the array
 * \param[in] device_array The device array
 * \param[in] count The number of elements of device_array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \return The same array allocated on the host
 * \note The source array might also be a managed array
 */
template <typename T>
T*
copyCreateDevice2HostArray(const T* device_array,
                           const stdgpu::index64_t count,
                           const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Creates and copies the given host array to the device
 * \tparam T The type of the array
 * \param[in] host_array The host array
 * \param[in] count The number of elements of host_array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \return The same array allocated on the device
 * \note The source array might also be a managed array
 */
template <typename T>
T*
copyCreateHost2DeviceArray(const T* host_array,
                           const stdgpu::index64_t count,
                           const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Creates and copies the given host array to the host
 * \tparam T The type of the array
 * \param[in] host_array The host array
 * \param[in] count The number of elements of host_array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \return The same array allocated on the host
 * \note The source array might also be a managed array
 */
template <typename T>
T*
copyCreateHost2HostArray(const T* host_array,
                         const stdgpu::index64_t count,
                         const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Creates and copies the given device array to the device
 * \tparam T The type of the array
 * \param[in] device_array The device array
 * \param[in] count The number of elements of device_array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \return The same array allocated on the device
 * \note The source array might also be a managed array
 */
template <typename T>
T*
copyCreateDevice2DeviceArray(const T* device_array,
                             const stdgpu::index64_t count,
                             const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Copies the given device array to the host
 * \tparam T The type of the array
 * \param[in] source_device_array The device array
 * \param[in] count The number of elements of source_device_array
 * \param[out] destination_host_array The host array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \note The source and destination arrays might also be managed arrays
 */
template <typename T>
void
copyDevice2HostArray(const T* source_device_array,
                     const stdgpu::index64_t count,
                     T* destination_host_array,
                     const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Copies the given host array to the device
 * \tparam T The type of the array
 * \param[in] source_host_array The host array
 * \param[in] count The number of elements of source_host_array
 * \param[out] destination_device_array The device array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \note The source and destination arrays might also be managed arrays
 */
template <typename T>
void
copyHost2DeviceArray(const T* source_host_array,
                     const stdgpu::index64_t count,
                     T* destination_device_array,
                     const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Copies the given host array to the host
 * \tparam T The type of the array
 * \param[in] source_host_array The host array
 * \param[in] count The number of elements of source_host_array
 * \param[out] destination_host_array The host array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \note The source and destination arrays might also be managed arrays
 */
template <typename T>
void
copyHost2HostArray(const T* source_host_array,
                   const stdgpu::index64_t count,
                   T* destination_host_array,
                   const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);


/**
 * \brief Copies the given device array to the device
 * \tparam T The type of the array
 * \param[in] source_device_array The device array
 * \param[in] count The number of elements of source_device_array
 * \param[out] destination_device_array The device array
 * \param[in] check_safety True if this function should check whether copying is safe, false otherwise
 * \note The source and destination arrays might also be managed arrays
 */
template <typename T>
void
copyDevice2DeviceArray(const T* source_device_array,
                       const stdgpu::index64_t count,
                       T* destination_device_array,
                       const MemoryCopy check_safety = MemoryCopy::RANGE_CHECK);



namespace stdgpu
{

/**
 * \brief The types of a dynamically allocated array
 */
enum class dynamic_memory_type
{
    host,           /**< The array is allocated on the host (CPU) */
    device,         /**< The array is allocated on the device (GPU) */
    managed,        /**< The array is allocated on both the host (CPU) and device (GPU) and managed internally by the driver via paging */
    invalid         /**< The array is not dynamically allocated by our API */
};


/**
 * \brief Determines the dynamic memory type of the given array
 * \param[in] array An array
 * \return The memory type of the array
 */
template <typename T>
dynamic_memory_type
get_dynamic_memory_type(T* array);


/**
 * \brief An allocator for device memory
 * \tparam T A type
 */
template <typename T>
struct safe_device_allocator
{
    using value_type = T;       /**< T */

    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::device;         /**< dynamic_memory_type::device */

    /**
     * \brief Allocates a memory block of the given size
     * \param[in] n The size of the memory block in bytes
     * \return A pointer to the allocated memory block
     */
    STDGPU_NODISCARD T*
    allocate(index64_t n);

    /**
     * \brief Deallocates the given memory block
     * \param[in] p A pointer to the memory block
     * \param[in] n The size of the memory block in bytes (must match the size during allocation)
     */
    void
    deallocate(T* p,
               index64_t n);
};


/**
 * \brief An allocator for pinned host memory
 * \tparam T A type
 */
template <typename T>
struct safe_pinned_host_allocator
{
    using value_type = T;       /**< T */

    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::host;           /**< dynamic_memory_type::host */

    /**
     * \brief Allocates a memory block of the given size
     * \param[in] n The size of the memory block in bytes
     * \return A pointer to the allocated memory block
     */
    STDGPU_NODISCARD T*
    allocate(index64_t n);

    /**
     * \brief Deallocates the given memory block
     * \param[in] p A pointer to the memory block
     * \param[in] n The size of the memory block in bytes (must match the size during allocation)
     */
    void
    deallocate(T* p,
               index64_t n);
};


/**
 * \brief An allocator for managed memory
 * \tparam T A type
 */
template <typename T>
struct safe_managed_allocator
{
    using value_type = T;       /**< T */

    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::managed;        /**< dynamic_memory_type::managed */

    /**
     * \brief Allocates a memory block of the given size
     * \param[in] n The size of the memory block in bytes
     * \return A pointer to the allocated memory block
     */
    STDGPU_NODISCARD T*
    allocate(index64_t n);

    /**
     * \brief Deallocates the given memory block
     * \param[in] p A pointer to the memory block
     * \param[in] n The size of the memory block in bytes (must match the size during allocation)
     */
    void
    deallocate(T* p,
               index64_t n);
};


/**
 * \brief A specialized default allocator traitor
 */
struct default_allocator_traits
{
    /**
     * \brief Constructs an object value at the given pointer
     * \tparam T The value type
     * \tparam Args The argument types
     * \param[in] p A pointer to the value
     * \param[in] args The arguments to construct the value
     */
    template <typename T, class... Args>
    static STDGPU_HOST_DEVICE void
    construct(T* p,
              Args&&... args);

    /**
     * \brief Destroys the object value at the given pointer
     * \tparam T The value type
     * \param[in] p A pointer to the value
     */
    template <typename T>
    static STDGPU_HOST_DEVICE void
    destroy(T* p);
};


/**
 * \brief Returns the total number of allocations of a specific memory type
 * \param[in] memory_type A dynamic memory type
 * \return The total number of allocation for the given type of memory if available, 0 otherwise
 */
index64_t
get_allocation_count(dynamic_memory_type memory_type);


/**
 * \brief Returns the total number of deallocations of a specific memory type
 * \param[in] memory_type A dynamic memory type
 * \return The total number of deallocation for the given type of memory if available, 0 otherwise
 */
index64_t
get_deallocation_count(dynamic_memory_type memory_type);


/**
 * \brief Finds the size (in bytes) of the given dynamically allocated array
 * \tparam T The type of the array
 * \param[in] array An array
 * \return The size (in bytes) of the given array if it was created by our API, 0 otherwise
 */
template <typename T>
index64_t
size_bytes(T* array);

} // namespace stdgpu



#include <stdgpu/impl/memory_detail.h>



#endif // STDGPU_MEMORY_H
