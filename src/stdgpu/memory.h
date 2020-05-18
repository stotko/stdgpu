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
 * \addtogroup memory memory
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/memory.h
 */

#include <memory>
#include <type_traits>

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
 * \ingroup memory
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
 * \ingroup memory
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

    /**
     * \brief Dynamic memory type of allocations
     */
    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::device;

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
 * \brief An allocator for host memory
 * \tparam T A type
 */
template <typename T>
struct safe_host_allocator
{
    using value_type = T;       /**< T */

    /**
     * \brief Dynamic memory type of allocations
     */
    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::host;

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

    /**
     * \brief Dynamic memory type of allocations
     */
    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::managed;

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
 * \brief A general allocator traitor
 *
 * Differences to std::allocator_traits:
 *  - No detection mechanism of the capabilities of the allocator, thus always using the fallback
 *  - index_type instead of size_type
 */
template <typename Allocator>
struct allocator_traits
{
    using allocator_type                            = Allocator;                                                                /**< Allocator */
    using value_type                                = typename Allocator::value_type;                                           /**< Allocator::value_type */
    using pointer                                   = value_type*;                                                              /**< value_type* */
    using const_pointer                             = typename std::pointer_traits<pointer>::template rebind<const value_type>; /**< std::pointer_traits<pointer>::rebind<const value_type> */
    using void_pointer                              = typename std::pointer_traits<pointer>::template rebind<void>;             /**< std::pointer_traits<pointer>::rebind<void> */
    using const_void_pointer                        = typename std::pointer_traits<pointer>::template rebind<const void>;       /**< std::pointer_traits<pointer>::rebind<const void> */
    using difference_type                           = typename std::pointer_traits<pointer>::difference_type;                   /**< std::pointer_traits<pointer>::difference_type */
    using index_type                                = index64_t;                                                                /**< index64_t */
    using propagate_on_container_copy_assignment    = std::false_type;                                                          /**< std::false_type */
    using propagate_on_container_move_assignment    = std::false_type;                                                          /**< std::false_type */
    using propagate_on_container_swap               = std::false_type;                                                          /**< std::false_type */
    using is_always_equal                           = std::is_empty<Allocator>;                                                 /**< std::is_empty<Allocator> */


    /**
     * \brief Allocates a memory block of the given size
     * \param[in] a The allocator to use
     * \param[in] n The size of the memory block in bytes
     * \return A pointer to the allocated memory block
     */
    STDGPU_NODISCARD static pointer
    allocate(Allocator& a,
             index_type n);

    /**
     * \brief Allocates a memory block of the given size
     * \param[in] a The allocator to use
     * \param[in] n The size of the memory block in bytes
     * \param[in] hint A pointer serving as a hint for the allocator
     * \return A pointer to the allocated memory block
     */
    STDGPU_NODISCARD static pointer
    allocate(Allocator& a,
             index_type n,
             const_void_pointer hint);

    /**
     * \brief Deallocates the given memory block
     * \param[in] a The allocator to use
     * \param[in] p A pointer to the memory block
     * \param[in] n The size of the memory block in bytes (must match the size during allocation)
     */
    static void
    deallocate(Allocator& a,
               pointer p,
               index_type n);

    /**
     * \brief Constructs an object value at the given pointer
     * \tparam T The value type
     * \tparam Args The argument types
     * \param[in] a The allocator to use
     * \param[in] p A pointer to the value
     * \param[in] args The arguments to construct the value
     */
    template <typename T, class... Args>
    static STDGPU_HOST_DEVICE void
    construct(Allocator& a,
              T* p,
              Args&&... args);

    /**
     * \brief Destroys the object value at the given pointer
     * \tparam T The value type
     * \param[in] a The allocator to use
     * \param[in] p A pointer to the value
     */
    template <typename T>
    static STDGPU_HOST_DEVICE void
    destroy(Allocator& a,
            T* p);

    /**
     * \brief Returns the maximum size that could be theoretically allocated
     * \param[in] a The allocator to use
     * \return The maximum size that could be theoretically allocated
     */
    static STDGPU_HOST_DEVICE index_type
    max_size(const Allocator& a);

    /**
     * \brief Returns a copy of the allocator
     * \param[in] a The allocator to use
     * \return A copy of the allocator
     */
    static Allocator
    select_on_container_copy_construction(const Allocator& a);
};


/**
 * \ingroup memory
 * \brief Destroys the value at the given pointer
 * \tparam T The value type
 * \tparam Args The argument types
 * \param[in] p A pointer to the value to construct
 * \param[in] args The arguments to construct the value
 * \return A pointer to the constructed value
 */
template <typename T, typename... Args>
STDGPU_HOST_DEVICE T*
construct_at(T* p,
             Args&&... args);


/**
 * \ingroup memory
 * \brief Destroys the value at the given pointer
 * \tparam T The value type
 * \param[in] p A pointer to the value to destroy
 */
template <typename T>
STDGPU_HOST_DEVICE void
destroy_at(T* p);


/**
 * \ingroup memory
 * \brief Destroys the range of values
 * \tparam Iterator The iterator type of the values
 * \param[in] first An iterator to the begin of the value range
 * \param[in] last An iterator to the end of the value range
 */
template <typename Iterator>
void
destroy(Iterator first,
        Iterator last);


/**
 * \ingroup memory
 * \brief Destroys the range of values
 * \tparam Iterator The iterator type of the values
 * \tparam Size The size type
 * \param[in] first An iterator to the begin of the value range
 * \param[in] n The number of elements in the value range
 * \return An iterator to the end of the value range
 */
template <typename Iterator, typename Size>
Iterator
destroy_n(Iterator first,
          Size n);


/**
 * \ingroup memory
 * \brief Returns the total number of allocations of a specific memory type
 * \param[in] memory_type A dynamic memory type
 * \return The total number of allocation for the given type of memory if available, 0 otherwise
 */
index64_t
get_allocation_count(dynamic_memory_type memory_type);


/**
 * \ingroup memory
 * \brief Returns the total number of deallocations of a specific memory type
 * \param[in] memory_type A dynamic memory type
 * \return The total number of deallocation for the given type of memory if available, 0 otherwise
 */
index64_t
get_deallocation_count(dynamic_memory_type memory_type);


/**
 * \ingroup memory
 * \brief Finds the size (in bytes) of the given dynamically allocated array
 * \tparam T The type of the array
 * \param[in] array An array
 * \return The size (in bytes) of the given array if it was created by our API, 0 otherwise
 */
template <typename T>
index64_t
size_bytes(T* array);


// Deprecated classes and functions
/**
 * \deprecated Replaced by stdgpu::safe_host_allocator<T>
 */
template <typename T>
struct safe_pinned_host_allocator;

/**
 * \deprecated Replaced by stdgpu::allocator_traits<Allocator>
 */
struct default_allocator_traits;

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/memory_detail.h>



#endif // STDGPU_MEMORY_H
