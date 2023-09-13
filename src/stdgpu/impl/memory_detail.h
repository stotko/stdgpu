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
#include <utility>

#include <stdgpu/algorithm.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/limits.h>
#include <stdgpu/platform.h>
#include <stdgpu/type_traits.h>
#include <stdgpu/utility.h>

namespace stdgpu::detail
{

template <typename T>
bool
is_destroy_optimizable()
{
    return std::is_trivially_destructible_v<T>;
}

[[nodiscard]] void*
allocate(index64_t bytes, dynamic_memory_type type);

void
deallocate(void* p, index64_t bytes, dynamic_memory_type type);

void
memcpy(void* destination,
       const void* source,
       index64_t bytes,
       dynamic_memory_type destination_type,
       dynamic_memory_type source_type,
       const bool external_memory);

template <typename Iterator, typename T>
class uninitialized_fill_functor
{
public:
    uninitialized_fill_functor(Iterator begin, const T& value) // NOLINT(modernize-pass-by-value)
      : _begin(begin)
      , _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index64_t i)
    {
        construct_at(to_address(_begin + i), _value);
    }

private:
    Iterator _begin;
    T _value;
};

template <typename InputIt, typename OutputIt>
class uninitialized_copy_functor
{
public:
    uninitialized_copy_functor(InputIt begin, OutputIt output_begin)
      : _begin(begin)
      , _output_begin(output_begin)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index64_t i)
    {
        // Directly dereference _begin instead of trying to get its raw pointer.
        // This adds support for transform_iterator, etc. which neither have operator->() nor get().
        construct_at(to_address(_output_begin + i), _begin[i]);
    }

private:
    InputIt _begin;
    OutputIt _output_begin;
};

template <typename Iterator>
class destroy_functor
{
public:
    explicit destroy_functor(Iterator first)
      : _first(first)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index64_t i)
    {
        destroy_at(to_address(_first + i));
    }

private:
    Iterator _first;
};

template <typename ExecutionPolicy, typename Iterator>
void
unoptimized_destroy(ExecutionPolicy&& policy, Iterator first, Iterator last)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   static_cast<index64_t>(last - first),
                   destroy_functor<Iterator>(first));
}

void
workaround_synchronize_managed_memory();

} // namespace stdgpu::detail

template <typename T>
T*
createDeviceArray(const stdgpu::index64_t count, const T default_value)
{
    T* device_array = nullptr;

#if STDGPU_DETAIL_IS_DEVICE_COMPILED
    using Allocator = stdgpu::safe_device_allocator<T>;
    Allocator device_allocator;

    device_array = stdgpu::allocator_traits<Allocator>::allocate_filled(stdgpu::execution::device,
                                                                        device_allocator,
                                                                        count,
                                                                        default_value);

    if (device_array == nullptr)
    {
        printf("createDeviceArray : Failed to allocate array. Aborting ...\n");
    }
#else
    T* host_array = createHostArray(count, default_value);

    device_array = copyCreateHost2DeviceArray(host_array, count);

    destroyHostArray(host_array);
#endif

    return device_array;
}

template <typename T>
T*
createHostArray(const stdgpu::index64_t count, const T default_value)
{
    using Allocator = stdgpu::safe_host_allocator<T>;
    Allocator host_allocator;

    T* host_array = stdgpu::allocator_traits<Allocator>::allocate_filled(stdgpu::execution::host,
                                                                         host_allocator,
                                                                         count,
                                                                         default_value);

    if (host_array == nullptr)
    {
        printf("createHostArray : Failed to allocate array. Aborting ...\n");
    }

    return host_array;
}

template <typename T>
T*
createManagedArray(const stdgpu::index64_t count, const T default_value, const Initialization initialize_on)
{
    using Allocator = stdgpu::safe_managed_allocator<T>;
    Allocator managed_allocator;

    T* managed_array = stdgpu::allocator_traits<Allocator>::allocate(managed_allocator, count);

    if (managed_array == nullptr)
    {
        printf("createManagedArray : Failed to allocate array. Aborting ...\n");
        return nullptr;
    }

    switch (initialize_on)
    {
#if STDGPU_DETAIL_IS_DEVICE_COMPILED
        case Initialization::DEVICE:
        {
            stdgpu::uninitialized_fill(stdgpu::execution::device,
                                       stdgpu::device_begin(managed_array),
                                       stdgpu::device_end(managed_array),
                                       default_value);
        }
        break;
#else
        case Initialization::DEVICE:
        {
            // Same as host path
        }
            [[fallthrough]];
#endif

        case Initialization::HOST:
        {
            stdgpu::detail::workaround_synchronize_managed_memory();

            stdgpu::uninitialized_fill(stdgpu::execution::host,
                                       stdgpu::host_begin(managed_array),
                                       stdgpu::host_end(managed_array),
                                       default_value);
        }
        break;

        default:
        {
            printf("createManagedArray : Invalid initialization device. Returning created but uninitialized array "
                   "...\n");
        }
    }

    return managed_array;
}

template <typename T>
void
destroyDeviceArray(T*& device_array)
{
    using Allocator = stdgpu::safe_device_allocator<T>;
    Allocator device_allocator;

#if STDGPU_DETAIL_IS_DEVICE_COMPILED
    stdgpu::allocator_traits<Allocator>::deallocate_filled(stdgpu::execution::device,
                                                           device_allocator,
                                                           device_array,
                                                           stdgpu::size(device_array));
    device_array = nullptr;
#else
    if (!stdgpu::detail::is_destroy_optimizable<T>())
    {
        T* host_array = copyCreateDevice2HostArray(device_array, stdgpu::size(device_array));

        // Calls destructor here
        destroyHostArray(host_array);
    }

    stdgpu::allocator_traits<Allocator>::deallocate(device_allocator, device_array, stdgpu::size(device_array));
    device_array = nullptr;
#endif
}

template <typename T>
void
destroyHostArray(T*& host_array)
{
    using Allocator = stdgpu::safe_host_allocator<T>;
    Allocator host_allocator;

    stdgpu::allocator_traits<Allocator>::deallocate_filled(stdgpu::execution::host,
                                                           host_allocator,
                                                           host_array,
                                                           stdgpu::size(host_array));
    host_array = nullptr;
}

template <typename T>
void
destroyManagedArray(T*& managed_array)
{
    using Allocator = stdgpu::safe_managed_allocator<T>;
    Allocator managed_allocator;

    // Call on host since the initialization place is not known
    stdgpu::allocator_traits<Allocator>::deallocate_filled(stdgpu::execution::host,
                                                           managed_allocator,
                                                           managed_array,
                                                           stdgpu::size(managed_array));
    managed_array = nullptr;
}

template <typename T>
T*
copyCreateDevice2HostArray(const T* device_array, const stdgpu::index64_t count, const MemoryCopy check_safety)
{
    using Allocator = stdgpu::safe_host_allocator<T>;
    Allocator host_allocator;

    T* host_array = stdgpu::allocator_traits<Allocator>::allocate(host_allocator, count);

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
copyCreateHost2DeviceArray(const T* host_array, const stdgpu::index64_t count, const MemoryCopy check_safety)
{
    using Allocator = stdgpu::safe_device_allocator<T>;
    Allocator device_allocator;

    T* device_array = stdgpu::allocator_traits<Allocator>::allocate(device_allocator, count);

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
copyCreateHost2HostArray(const T* host_array, const stdgpu::index64_t count, const MemoryCopy check_safety)
{
    using Allocator = stdgpu::safe_host_allocator<T>;
    Allocator host_allocator;

    T* host_array_2 = stdgpu::allocator_traits<Allocator>::allocate(host_allocator, count);

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
copyCreateDevice2DeviceArray(const T* device_array, const stdgpu::index64_t count, const MemoryCopy check_safety)
{
    using Allocator = stdgpu::safe_device_allocator<T>;
    Allocator device_allocator;

    T* device_array_2 = stdgpu::allocator_traits<Allocator>::allocate(device_allocator, count);

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
                           count * static_cast<stdgpu::index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
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
                           count * static_cast<stdgpu::index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
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
                           count * static_cast<stdgpu::index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
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
                           count * static_cast<stdgpu::index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                           stdgpu::dynamic_memory_type::device,
                           stdgpu::dynamic_memory_type::device,
                           check_safety != MemoryCopy::RANGE_CHECK);
}

namespace stdgpu
{

template <typename T>
device_unique_object<T>::device_unique_object(null_object_t /*null_object*/)
  : _object(nullptr)
{
}

template <typename T>
template <typename... Args>
device_unique_object<T>::device_unique_object(Args&&... args)
  : _object(new T(T::createDeviceObject(std::forward<Args>(args)...)), [](T* ptr) {
      T::destroyDeviceObject(*ptr);
      delete ptr;
  })
{
}

template <typename T>
template <typename ExecutionPolicy,
          typename... Args,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
device_unique_object<T>::device_unique_object(ExecutionPolicy&& policy, Args&&... args)
  : _object(new T(T::createDeviceObject(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...)),
            [_policy = std::forward<ExecutionPolicy>(policy)](T* ptr) {
                T::destroyDeviceObject(_policy, *ptr);
                delete ptr;
            })
{
}

template <typename T>
const T*
device_unique_object<T>::operator->() const
{
    return _object.operator->();
}

template <typename T>
T*
device_unique_object<T>::operator->()
{
    return _object.operator->();
}

template <typename T>
const T&
device_unique_object<T>::operator*() const
{
    return *_object;
}

template <typename T>
T&
device_unique_object<T>::operator*()
{
    return *_object;
}

template <typename T>
device_unique_object<T>::operator bool() const
{
    return _object.operator bool();
}

template <typename T>
template <typename U>
safe_device_allocator<T>::safe_device_allocator([[maybe_unused]] const safe_device_allocator<U>& other) noexcept
{
}

template <typename T>
[[nodiscard]] T*
safe_device_allocator<T>::allocate(index64_t n)
{
    T* p = static_cast<T*>(
            detail::allocate(n * static_cast<index64_t>(sizeof(T)), memory_type)); // NOLINT(bugprone-sizeof-expression)
    register_memory(p, n, memory_type);
    return p;
}

template <typename T>
void
safe_device_allocator<T>::deallocate(T* p, index64_t n)
{
    deregister_memory(p, n, memory_type);
    detail::deallocate(static_cast<void*>(p),
                       n * static_cast<index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                       memory_type);
}

template <typename T>
template <typename U>
safe_host_allocator<T>::safe_host_allocator([[maybe_unused]] const safe_host_allocator<U>& other) noexcept
{
}

template <typename T>
[[nodiscard]] T*
safe_host_allocator<T>::allocate(index64_t n)
{
    T* p = static_cast<T*>(
            detail::allocate(n * static_cast<index64_t>(sizeof(T)), memory_type)); // NOLINT(bugprone-sizeof-expression)
    register_memory(p, n, memory_type);
    return p;
}

template <typename T>
void
safe_host_allocator<T>::deallocate(T* p, index64_t n)
{
    deregister_memory(p, n, memory_type);
    detail::deallocate(static_cast<void*>(p),
                       n * static_cast<index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                       memory_type);
}

template <typename T>
template <typename U>
safe_managed_allocator<T>::safe_managed_allocator([[maybe_unused]] const safe_managed_allocator<U>& other) noexcept
{
}

template <typename T>
[[nodiscard]] T*
safe_managed_allocator<T>::allocate(index64_t n)
{
    T* p = static_cast<T*>(
            detail::allocate(n * static_cast<index64_t>(sizeof(T)), memory_type)); // NOLINT(bugprone-sizeof-expression)
    register_memory(p, n, memory_type);
    return p;
}

template <typename T>
void
safe_managed_allocator<T>::deallocate(T* p, index64_t n)
{
    deregister_memory(p, n, memory_type);
    detail::deallocate(static_cast<void*>(p),
                       n * static_cast<index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                       memory_type);
}

template <typename Allocator>
typename allocator_traits<Allocator>::pointer
allocator_traits<Allocator>::allocate(Allocator& a, typename allocator_traits<Allocator>::index_type n)
{
    return a.allocate(n);
}

template <typename Allocator>
typename allocator_traits<Allocator>::pointer
allocator_traits<Allocator>::allocate(Allocator& a,
                                      typename allocator_traits<Allocator>::index_type n,
                                      // cppcheck-suppress syntaxError
                                      [[maybe_unused]] typename allocator_traits<Allocator>::const_void_pointer hint)
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
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
typename allocator_traits<Allocator>::pointer
allocator_traits<Allocator>::allocate_filled(ExecutionPolicy&& policy,
                                             Allocator& a,
                                             typename allocator_traits<Allocator>::index_type n,
                                             const typename allocator_traits<Allocator>::value_type& default_value)
{
    pointer p = allocate(a, n);
    if (p != nullptr)
    {
        stdgpu::uninitialized_fill(std::forward<ExecutionPolicy>(policy), p, p + size(p), default_value);
    }
    return p;
}

template <typename Allocator>
template <typename ExecutionPolicy,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
allocator_traits<Allocator>::deallocate_filled(ExecutionPolicy&& policy,
                                               Allocator& a,
                                               typename allocator_traits<Allocator>::pointer p,
                                               typename allocator_traits<Allocator>::index_type n)
{
    if (!detail::is_destroy_optimizable<value_type>())
    {
        stdgpu::destroy(std::forward<ExecutionPolicy>(policy), p, p + size(p));
    }
    return deallocate(a, p, n);
}

template <typename Allocator>
template <typename T, class... Args>
STDGPU_HOST_DEVICE void
allocator_traits<Allocator>::construct([[maybe_unused]] Allocator& a, T* p, Args&&... args)
{
    construct_at(p, forward<Args>(args)...);
}

template <typename Allocator>
template <typename T>
STDGPU_HOST_DEVICE void
allocator_traits<Allocator>::destroy([[maybe_unused]] Allocator& a, T* p)
{
    destroy_at(p);
}

template <typename Allocator>
STDGPU_HOST_DEVICE typename allocator_traits<Allocator>::index_type
allocator_traits<Allocator>::max_size([[maybe_unused]] const Allocator& a) noexcept
{
    return numeric_limits<index_type>::max() / sizeof(value_type);
}

template <typename Allocator>
Allocator
allocator_traits<Allocator>::select_on_container_copy_construction(const Allocator& a)
{
    return a;
}

template <typename T>
STDGPU_HOST_DEVICE T*
to_address(T* p) noexcept
{
    return p;
}

// Use pre-C++17 SFINAE for dispatching due to wrong missing-return warning caused by NVCC
// (potentially fixed in CUDA 11.5+)
/*
template <typename Ptr>
STDGPU_HOST_DEVICE auto
to_address(const Ptr& p) noexcept
{
    if constexpr (detail::has_arrow_operator_v<Ptr>)
    {
        return to_address(p.operator->());
    }
    else if constexpr (!detail::has_arrow_operator_v<Ptr> && detail::has_get_v<Ptr>)
    {
        return to_address(p.get());
    }
    else
    {
        static_assert(detail::dependent_false_v<Ptr>, "Ptr has neither operator->() or get() defined.");

        // This reduces the number of compiler errors in calling contexts and makes the failed assertion more apparent.
        return static_cast<void*>(nullptr);
    }
}
*/

template <typename Ptr, STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(detail::has_arrow_operator_v<Ptr>)>
STDGPU_HOST_DEVICE auto
to_address(const Ptr& p) noexcept
{
    return to_address(p.operator->());
}

template <typename Ptr,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(!detail::has_arrow_operator_v<Ptr> && detail::has_get_v<Ptr>)>
STDGPU_HOST_DEVICE auto
to_address(const Ptr& p) noexcept
{
    return to_address(p.get());
}

template <typename T, typename... Args>
STDGPU_HOST_DEVICE T*
construct_at(T* p, Args&&... args)
{
    return ::new (static_cast<void*>(p)) T(forward<Args>(args)...);
}

template <typename T>
STDGPU_HOST_DEVICE void
destroy_at(T* p)
{
    p->~T();
}

template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
uninitialized_fill(ExecutionPolicy&& policy, Iterator begin, Iterator end, const T& value)
{
    uninitialized_fill_n(std::forward<ExecutionPolicy>(policy), begin, static_cast<index64_t>(end - begin), value);
}

template <typename ExecutionPolicy,
          typename Iterator,
          typename Size,
          typename T,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
Iterator
uninitialized_fill_n(ExecutionPolicy&& policy, Iterator begin, Size n, const T& value)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   n,
                   detail::uninitialized_fill_functor<Iterator, T>(begin, value));
    return begin + n;
}

template <typename ExecutionPolicy,
          typename InputIt,
          typename OutputIt,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
OutputIt
uninitialized_copy(ExecutionPolicy&& policy, InputIt begin, InputIt end, OutputIt output_begin)
{
    return uninitialized_copy_n(std::forward<ExecutionPolicy>(policy),
                                begin,
                                static_cast<index64_t>(end - begin),
                                output_begin);
}

template <typename ExecutionPolicy,
          typename InputIt,
          typename Size,
          typename OutputIt,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
OutputIt
uninitialized_copy_n(ExecutionPolicy&& policy, InputIt begin, Size n, OutputIt output_begin)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   n,
                   detail::uninitialized_copy_functor<InputIt, OutputIt>(begin, output_begin));
    return output_begin + n;
}

template <typename ExecutionPolicy,
          typename Iterator,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
destroy(ExecutionPolicy&& policy, Iterator first, Iterator last)
{
    if (!detail::is_destroy_optimizable<typename std::iterator_traits<Iterator>::value_type>())
    {
        detail::unoptimized_destroy(std::forward<ExecutionPolicy>(policy), first, last);
    }
}

template <typename ExecutionPolicy,
          typename Iterator,
          typename Size,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
Iterator
destroy_n(ExecutionPolicy&& policy, Iterator first, Size n)
{
    Iterator last = first + n;

    destroy(std::forward<ExecutionPolicy>(policy), first, last);

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
void
register_memory(void* p, index64_t n, dynamic_memory_type memory_type);

template <typename T>
void
register_memory(T* p, index64_t n, dynamic_memory_type memory_type)
{
    register_memory<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(p)),
                          n * static_cast<index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                          memory_type);
}

template <>
void
deregister_memory(void* p, index64_t n, dynamic_memory_type memory_type);

template <typename T>
void
deregister_memory(T* p, index64_t n, dynamic_memory_type memory_type)
{
    deregister_memory<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(p)),
                            n * static_cast<index64_t>(sizeof(T)), // NOLINT(bugprone-sizeof-expression)
                            memory_type);
}

template <>
index64_t
size_bytes(void* array);

template <typename T>
index64_t
size_bytes(T* array)
{
    return size_bytes<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(array)));
}

} // namespace stdgpu

#endif // STDGPU_MEMORY_DETAIL_H
