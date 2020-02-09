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

#ifndef STDGPU_MEMORY_DETAIL_H
#define STDGPU_MEMORY_DETAIL_H

#include <cstdio>
#include <type_traits>
#include <thrust/for_each.h>

#include <stdgpu/attribute.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/limits.h>
#include <stdgpu/platform.h>
#include <stdgpu/utility.h>



namespace stdgpu
{

namespace detail
{

STDGPU_NODISCARD void*
allocate(index64_t bytes,
         dynamic_memory_type type);

void
deallocate(void* p,
           index64_t bytes,
           dynamic_memory_type type);

void
memcpy(void* destination,
       const void* source,
       index64_t bytes,
       dynamic_memory_type destination_type,
       dynamic_memory_type source_type,
       const bool external_memory);

template <typename T>
class construct_value
{
    public:
        STDGPU_HOST_DEVICE
        construct_value(const T& value)
            : _value(value)
        {

        }

        STDGPU_HOST_DEVICE void
        operator()(T& t) const
        {
            construct_at(&t, _value);
        }

    private:
        T _value;
};

template <typename Iterator, typename T>
void
uninitialized_fill(Iterator begin,
                   Iterator end,
                   const T& value)
{
    // Define own version as thrust uses an optimization too aggressively which causes compilation failures for certain types
    thrust::for_each(begin, end,
                     construct_value<T>(value));
}

template <typename T>
struct destroy_value
{
    STDGPU_HOST_DEVICE void
    operator()(T& t) const
    {
        destroy_at(&t);
    }
};

void
workaround_synchronize_device_thrust();

void
workaround_synchronize_managed_memory();

} // namespace detail

} // namespace stdgpu



template <typename T>
T*
createDeviceArray(const stdgpu::index64_t count,
                  const T default_value)
{
    T* device_array = nullptr;

    #if STDGPU_BACKEND != STDGPU_BACKEND_CUDA || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
        stdgpu::safe_device_allocator<T> device_allocator;
        device_array = device_allocator.allocate(count);

        if (device_array == nullptr)
        {
            printf("createDeviceArray : Failed to allocate array. Aborting ...\n");
            return nullptr;
        }

        stdgpu::detail::uninitialized_fill(stdgpu::device_begin(device_array), stdgpu::device_end(device_array),
                                           default_value);

        stdgpu::detail::workaround_synchronize_device_thrust();
    #else
        #if STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING
            printf("createDeviceArray : Creating auxiliary array on host to enable execution on host compiler ...\n");
        #endif

        T* host_array = createHostArray(count, default_value);

        device_array = copyCreateHost2DeviceArray(host_array, count);

        destroyHostArray(host_array);
    #endif

    return device_array;
}


template <typename T>
T*
createHostArray(const stdgpu::index64_t count,
                const T default_value)
{
    T* host_array = nullptr;

    stdgpu::safe_host_allocator<T> host_allocator;
    host_array = host_allocator.allocate(count);

    if (host_array == nullptr)
    {
        printf("createHostArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    stdgpu::detail::uninitialized_fill(stdgpu::host_begin(host_array), stdgpu::host_end(host_array),
                                       default_value);

    return host_array;
}


template <typename T>
T*
createManagedArray(const stdgpu::index64_t count,
                   const T default_value,
                   const Initialization initialize_on)
{
    T* managed_array = nullptr;

    stdgpu::safe_managed_allocator<T> managed_allocator;
    managed_array = managed_allocator.allocate(count);

    if (managed_array == nullptr)
    {
        printf("createManagedArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    switch (initialize_on)
    {
        #if STDGPU_BACKEND != STDGPU_BACKEND_CUDA || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
            case Initialization::DEVICE :
            {
                stdgpu::detail::uninitialized_fill(stdgpu::device_begin(managed_array), stdgpu::device_end(managed_array),
                                                   default_value);

                stdgpu::detail::workaround_synchronize_device_thrust();
            }
            break;
        #else
            case Initialization::DEVICE : // Same as host path
            {
                #if STDGPU_ENABLE_MANAGED_ARRAY_WARNING
                    printf("createManagedArray : Setting default value on host to enable execution on host compiler ...\n");
                #endif
            }
            STDGPU_FALLTHROUGH;
        #endif

        case Initialization::HOST :
        {
            stdgpu::detail::workaround_synchronize_managed_memory();

            stdgpu::detail::uninitialized_fill(stdgpu::host_begin(managed_array), stdgpu::host_end(managed_array),
                                               default_value);
        }
        break;

        default :
        {
            printf("createManagedArray : Invalid initialization device. Returning created but uninitialized array ...\n");
        }
    }

    return managed_array;
}


template <typename T>
void
destroyDeviceArray(T*& device_array)
{
    #if !STDGPU_USE_FAST_DESTROY
        #if STDGPU_BACKEND != STDGPU_BACKEND_CUDA || STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_NVCC
            stdgpu::destroy(stdgpu::device_begin(device_array), stdgpu::device_end(device_array));

            stdgpu::detail::workaround_synchronize_device_thrust();
        #else
            #if STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING
                printf("destroyDeviceArray : Creating auxiliary array on host to enable execution on host compiler ...\n");
            #endif

            T* host_array = copyCreateDevice2HostArray(device_array, stdgpu::size(device_array));

            // Calls destructor here
            destroyHostArray(host_array);
        #endif
    #endif

    stdgpu::safe_device_allocator<T> device_allocator;
    device_allocator.deallocate(device_array, stdgpu::size(device_array));

    device_array = nullptr;
}


template <typename T>
void
destroyHostArray(T*& host_array)
{
    #if !STDGPU_USE_FAST_DESTROY
        stdgpu::destroy(stdgpu::host_begin(host_array), stdgpu::host_end(host_array));
    #endif

    stdgpu::safe_host_allocator<T> host_allocator;
    host_allocator.deallocate(host_array, stdgpu::size(host_array));

    host_array = nullptr;
}


template <typename T>
void
destroyManagedArray(T*& managed_array)
{
    #if !STDGPU_USE_FAST_DESTROY
        // Call on host since the initialization place is not known
        stdgpu::destroy(stdgpu::host_begin(managed_array), stdgpu::host_end(managed_array));
    #endif

    stdgpu::safe_managed_allocator<T> managed_allocator;
    managed_allocator.deallocate(managed_array, stdgpu::size(managed_array));

    managed_array = nullptr;
}


template <typename T>
T*
copyCreateDevice2HostArray(const T* device_array,
                           const stdgpu::index64_t count,
                           const MemoryCopy check_safety)
{
    T* host_array = nullptr;

    stdgpu::safe_host_allocator<T> host_allocator;
    host_array = host_allocator.allocate(count);

    if (host_array == nullptr)
    {
        printf("copyCreateDevice2HostArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    copyDevice2HostArray(device_array, count, host_array, check_safety);

    return host_array;
}


template <typename T>
T*
copyCreateHost2DeviceArray(const T* host_array,
                           const stdgpu::index64_t count,
                           const MemoryCopy check_safety)
{
    T* device_array = nullptr;

    stdgpu::safe_device_allocator<T> device_allocator;
    device_array = device_allocator.allocate(count);

    if (device_array == nullptr)
    {
        printf("copyCreateHost2DeviceArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    copyHost2DeviceArray(host_array, count, device_array, check_safety);

    return device_array;
}


template <typename T>
T*
copyCreateHost2HostArray(const T* host_array,
                         const stdgpu::index64_t count,
                         const MemoryCopy check_safety)
{
    T* host_array_2 = nullptr;

    stdgpu::safe_host_allocator<T> host_allocator;
    host_array_2 = host_allocator.allocate(count);

    if (host_array_2 == nullptr)
    {
        printf("copyCreateHost2HostArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    copyHost2HostArray(host_array, count, host_array_2, check_safety);

    return host_array_2;
}


template <typename T>
T*
copyCreateDevice2DeviceArray(const T* device_array,
                             const stdgpu::index64_t count,
                             const MemoryCopy check_safety)
{
    T* device_array_2 = nullptr;

    stdgpu::safe_device_allocator<T> device_allocator;
    device_array_2 = device_allocator.allocate(count);

    if (device_array_2 == nullptr)
    {
        printf("copyCreateDevice2DeviceArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    copyDevice2DeviceArray(device_array, count, device_array_2, check_safety);

    return device_array_2;
}



template <typename T>
void
copyDevice2HostArray(const T* source_device_array,
                     const stdgpu::index64_t count,
                     T* destination_host_array,
                     const MemoryCopy check_safety)
{
    stdgpu::detail::memcpy(destination_host_array,
                           source_device_array,
                           count * sizeof(T),
                           stdgpu::dynamic_memory_type::host,
                           stdgpu::dynamic_memory_type::device,
                           check_safety != MemoryCopy::RANGE_CHECK);
}


template <typename T>
void
copyHost2DeviceArray(const T* source_host_array,
                     const stdgpu::index64_t count,
                     T* destination_device_array,
                     const MemoryCopy check_safety)
{
    stdgpu::detail::memcpy(destination_device_array,
                           source_host_array,
                           count * sizeof(T),
                           stdgpu::dynamic_memory_type::device,
                           stdgpu::dynamic_memory_type::host,
                           check_safety != MemoryCopy::RANGE_CHECK);
}


template <typename T>
void
copyHost2HostArray(const T* source_host_array,
                   const stdgpu::index64_t count,
                   T* destination_host_array,
                   const MemoryCopy check_safety)
{
    stdgpu::detail::memcpy(destination_host_array,
                           source_host_array,
                           count * sizeof(T),
                           stdgpu::dynamic_memory_type::host,
                           stdgpu::dynamic_memory_type::host,
                           check_safety != MemoryCopy::RANGE_CHECK);
}


template <typename T>
void
copyDevice2DeviceArray(const T* source_device_array,
                       const stdgpu::index64_t count,
                       T* destination_device_array,
                       const MemoryCopy check_safety)
{
    stdgpu::detail::memcpy(destination_device_array,
                           source_device_array,
                           count * sizeof(T),
                           stdgpu::dynamic_memory_type::device,
                           stdgpu::dynamic_memory_type::device,
                           check_safety != MemoryCopy::RANGE_CHECK);
}



namespace stdgpu
{

template <typename T>
STDGPU_NODISCARD T*
safe_device_allocator<T>::allocate(index64_t n)
{
    return static_cast<T*>(detail::allocate(n * sizeof(T), memory_type));
}


template <typename T>
void
safe_device_allocator<T>::deallocate(T* p,
                                     index64_t n)
{
    detail::deallocate(static_cast<void*>(p), n * sizeof(T), memory_type);
}


template <typename T>
STDGPU_NODISCARD T*
safe_host_allocator<T>::allocate(index64_t n)
{
    return static_cast<T*>(detail::allocate(n * sizeof(T), memory_type));
}


template <typename T>
void
safe_host_allocator<T>::deallocate(T* p,
                                   index64_t n)
{
    detail::deallocate(static_cast<void*>(p), n * sizeof(T), memory_type);
}


template <typename T>
STDGPU_NODISCARD T*
safe_managed_allocator<T>::allocate(index64_t n)
{
    return static_cast<T*>(detail::allocate(n * sizeof(T), memory_type));
}


template <typename T>
void
safe_managed_allocator<T>::deallocate(T* p,
                                      index64_t n)
{
    detail::deallocate(static_cast<void*>(p), n * sizeof(T), memory_type);
}


template <typename Allocator>
typename allocator_traits<Allocator>::pointer
allocator_traits<Allocator>::allocate(Allocator& a,
                                      typename allocator_traits<Allocator>::index_type n)
{
    return a.allocate(n);
}


template <typename Allocator>
typename allocator_traits<Allocator>::pointer
allocator_traits<Allocator>::allocate(Allocator& a,
                                      typename allocator_traits<Allocator>::index_type n,
                                      STDGPU_MAYBE_UNUSED typename allocator_traits<Allocator>::const_void_pointer hint)
{
    return a.allocate(n);
}


template <typename Allocator>
void
allocator_traits<Allocator>::deallocate(Allocator& a,
                                        typename allocator_traits<Allocator>::pointer p,
                                        typename allocator_traits<Allocator>::index_type n)
{
    return a.deallocate(p, n);
}


template <typename Allocator>
template <typename T, class... Args>
STDGPU_HOST_DEVICE void
allocator_traits<Allocator>::construct(STDGPU_MAYBE_UNUSED Allocator& a,
                                       T* p,
                                       Args&&... args)
{
    construct_at(p, forward<Args>(args)...);
}


template <typename Allocator>
template <typename T>
STDGPU_HOST_DEVICE void
allocator_traits<Allocator>::destroy(STDGPU_MAYBE_UNUSED Allocator& a,
                                     T* p)
{
    destroy_at(p);
}


template <typename Allocator>
STDGPU_HOST_DEVICE typename allocator_traits<Allocator>::index_type
allocator_traits<Allocator>::max_size(STDGPU_MAYBE_UNUSED const Allocator& a)
{
    return stdgpu::numeric_limits<index_type>::max() / sizeof(value_type);
}


template <typename Allocator>
Allocator
allocator_traits<Allocator>::select_on_container_copy_construction(STDGPU_MAYBE_UNUSED const Allocator& a)
{
    return a;
}


template <typename T, typename... Args>
STDGPU_HOST_DEVICE T*
construct_at(T* p,
             Args&&... args)
{
    return ::new (static_cast<void*>(p)) T(forward<Args>(args)...);
}


template <typename T>
STDGPU_HOST_DEVICE void
destroy_at(T* p)
{
    p->~T();
}


template <typename Iterator>
void
destroy(Iterator first,
        Iterator last)
{
    thrust::for_each(first, last,
                     detail::destroy_value<typename std::iterator_traits<Iterator>::value_type>());
}


template <typename Iterator, typename Size>
Iterator
destroy_n(Iterator first,
          Size n)
{
    Iterator last = first + n;

    destroy(first, last);

    return last;
}


template <>
dynamic_memory_type
get_dynamic_memory_type(void* array);


template <typename T>
dynamic_memory_type
get_dynamic_memory_type(T* array)
{
    return get_dynamic_memory_type<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(array)));
}


template <>
stdgpu::index64_t
size_bytes(void* array);


template <typename T>
index64_t
size_bytes(T* array)
{
    return size_bytes<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(array)));
}


// Deprecated classes and functions
template <typename T>
struct [[deprecated("Replaced by stdgpu::safe_host_allocator<T>")]] safe_pinned_host_allocator
{
    using value_type = T;

    constexpr static dynamic_memory_type memory_type = dynamic_memory_type::host;

    STDGPU_NODISCARD T*
    allocate(index64_t n)
    {
        return static_cast<T*>(detail::allocate(n * sizeof(T), memory_type));
    }

    void
    deallocate(T* p,
               index64_t n)
    {
        detail::deallocate(static_cast<void*>(p), n * sizeof(T), memory_type);
    }
};

struct [[deprecated("Replaced by stdgpu::allocator_traits<Allocator>")]] default_allocator_traits
{
    template <typename T, class... Args>
    static STDGPU_HOST_DEVICE void
    construct(T* p,
              Args&&... args)
    {
        construct_at(p, forward<Args>(args)...);
    }

    template <typename T>
    static STDGPU_HOST_DEVICE void
    destroy(T* p)
    {
        destroy_at(p);
    }
};

} // namespace stdgpu



#endif // STDGPU_MEMORY_DETAIL_H
