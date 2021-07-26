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

#ifndef STDGPU_ATOMIC_DETAIL_H
#define STDGPU_ATOMIC_DETAIL_H

#include <stdgpu/config.h>

#if STDGPU_BACKEND == STDGPU_BACKEND_CUDA
    #define STDGPU_BACKEND_ATOMIC_HEADER <stdgpu/STDGPU_BACKEND_DIRECTORY/atomic.cuh> // NOLINT(bugprone-macro-parentheses,misc-macro-parentheses)
    // cppcheck-suppress preprocessorErrorDirective
    #include STDGPU_BACKEND_ATOMIC_HEADER
    #undef STDGPU_BACKEND_ATOMIC_HEADER
#else
    #define STDGPU_BACKEND_ATOMIC_HEADER <stdgpu/STDGPU_BACKEND_DIRECTORY/atomic.h> // NOLINT(bugprone-macro-parentheses,misc-macro-parentheses)
    // cppcheck-suppress preprocessorErrorDirective
    #include STDGPU_BACKEND_ATOMIC_HEADER
    #undef STDGPU_BACKEND_ATOMIC_HEADER
#endif

#include <stdgpu/attribute.h>
#include <stdgpu/memory.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

namespace detail
{
    inline STDGPU_DEVICE_ONLY void
    atomic_load_thread_fence(const memory_order order)
    {
        switch (order)
        {
            case memory_order_consume :
            case memory_order_acquire :
            case memory_order_acq_rel :
            case memory_order_seq_cst :
            {
                stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_thread_fence();
            }
            break;

            case memory_order_relaxed :
            case memory_order_release :
            default :
            {
                // Nothing to do ...
            }
        }
    }

    inline STDGPU_DEVICE_ONLY void
    atomic_store_thread_fence(const memory_order order)
    {
        switch (order)
        {
            case memory_order_release :
            case memory_order_acq_rel :
            case memory_order_seq_cst :
            {
                stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_thread_fence();
            }
            break;

            case memory_order_relaxed :
            case memory_order_consume :
            case memory_order_acquire :
            default :
            {
                // Nothing to do ...
            }
        }
    }

    inline STDGPU_DEVICE_ONLY void
    atomic_consistency_thread_fence(const memory_order order)
    {
        switch (order)
        {
            case memory_order_seq_cst :
            {
                stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_thread_fence();
            }
            break;

            case memory_order_relaxed :
            case memory_order_consume :
            case memory_order_acquire :
            case memory_order_release :
            case memory_order_acq_rel :
            default :
            {
                // Nothing to do ...
            }
        }
    }
} // namespace detail


inline STDGPU_DEVICE_ONLY void
atomic_thread_fence(const memory_order order)
{
    switch (order)
    {
        case memory_order_consume :
        case memory_order_acquire :
        case memory_order_release :
        case memory_order_acq_rel :
        case memory_order_seq_cst :
        {
            stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_thread_fence();
        }
        break;

        case memory_order_relaxed :
        default :
        {
            // Nothing to do ...
        }
    }
}


inline STDGPU_DEVICE_ONLY void
atomic_signal_fence(const memory_order order)
{
    atomic_thread_fence(order);
}


template <typename T, typename Allocator>
inline atomic<T, Allocator>
atomic<T, Allocator>::createDeviceObject(const Allocator& allocator)
{
    atomic<T, Allocator> result(allocator);
    result._value_ref._value = createDeviceArray<T, allocator_type>(result._allocator, 1, 0);

    return result;
}


template <typename T, typename Allocator>
inline void
atomic<T, Allocator>::destroyDeviceObject(atomic<T, Allocator>& device_object)
{
    destroyDeviceArray<T, allocator_type>(device_object._allocator, device_object._value_ref._value);
}


template <typename T, typename Allocator>
inline
atomic<T, Allocator>::atomic()
    : _value_ref(nullptr)
{

}


template <typename T, typename Allocator>
inline
atomic<T, Allocator>::atomic(const Allocator& allocator)
    : _value_ref(nullptr),
      _allocator(allocator)
{

}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE typename atomic<T, Allocator>::allocator_type
atomic<T, Allocator>::get_allocator() const
{
    return _allocator;
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
atomic<T, Allocator>::is_lock_free() const
{
    return _value_ref.is_lock_free();
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE T
atomic<T, Allocator>::load(const memory_order order) const
{
    return _value_ref.load(order);
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE
atomic<T, Allocator>::operator T() const
{
    return _value_ref.operator T();
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE void
atomic<T, Allocator>::store(const T desired,
                            const memory_order order)
{
    _value_ref.store(desired, order);
}


template <typename T, typename Allocator> //NOLINT(misc-unconventional-assign-operator)
inline STDGPU_HOST_DEVICE T //NOLINT(misc-unconventional-assign-operator)
atomic<T, Allocator>::operator=(const T desired)
{
    return _value_ref.operator=(desired);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::exchange(const T desired,
                    const memory_order order)
{
    return _value_ref.exchange(desired, order);
}



template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
atomic<T, Allocator>::compare_exchange_weak(T& expected,
                                            const T desired,
                                            const memory_order order)
{
    return _value_ref.compare_exchange_weak(expected, desired, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
atomic<T, Allocator>::compare_exchange_strong(T& expected,
                                              const T desired,
                                              const memory_order order)
{
    return _value_ref.compare_exchange_strong(expected, desired, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_add(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_add(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_sub(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_sub(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_and(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_and(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_or(const T arg,
                               const memory_order order)
{
    return _value_ref.fetch_or(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_xor(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_xor(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_min(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_min(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_max(const T arg,
                                const memory_order order)
{
    return _value_ref.fetch_max(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same<T, unsigned int>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_inc_mod(const T arg,
                                    const memory_order order)
{
    return _value_ref.fetch_inc_mod(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same<T, unsigned int>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::fetch_dec_mod(const T arg,
                                    const memory_order order)
{
    return _value_ref.fetch_dec_mod(arg, order);
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator++()
{
    return ++_value_ref;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator++(int)
{
    return _value_ref++;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator--()
{
    return --_value_ref;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator--(int)
{
    return _value_ref--;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator+=(const T arg)
{
    return _value_ref += arg;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator-=(const T arg)
{
    return _value_ref -= arg;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator&=(const T arg)
{
    return _value_ref &= arg;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator|=(const T arg)
{
    return _value_ref |= arg;
}


template <typename T, typename Allocator>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic<T, Allocator>::operator^=(const T arg)
{
    return _value_ref ^= arg;
}



template <typename T>
inline STDGPU_HOST_DEVICE
atomic_ref<T>::atomic_ref(T& obj)
{
    _value = &obj;
}


template <typename T>
inline STDGPU_HOST_DEVICE
atomic_ref<T>::atomic_ref(T* value)
{
    _value = value;
}


template <typename T>
inline STDGPU_HOST_DEVICE bool
atomic_ref<T>::is_lock_free() const
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_is_lock_free();
}


template <typename T>
inline STDGPU_HOST_DEVICE T
atomic_ref<T>::load(STDGPU_MAYBE_UNUSED const memory_order order) const
{
    if (_value == nullptr)
    {
        return 0;
    }

    T local_value;
    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        detail::atomic_load_thread_fence(order);

        local_value = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_load(_value);

        detail::atomic_consistency_thread_fence(order);
    #else
        copyDevice2HostArray<T>(_value, 1, &local_value, MemoryCopy::NO_CHECK);
    #endif

    return local_value;
}


template <typename T>
inline STDGPU_HOST_DEVICE
atomic_ref<T>::operator T() const
{
    return load();
}


template <typename T>
inline STDGPU_HOST_DEVICE void
atomic_ref<T>::store(const T desired,
                     STDGPU_MAYBE_UNUSED const memory_order order)
{
    if (_value == nullptr)
    {
        return;
    }

    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        detail::atomic_consistency_thread_fence(order);

        stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_store(_value, desired);

        detail::atomic_store_thread_fence(order);
    #else
        copyHost2DeviceArray<T>(&desired, 1, _value, MemoryCopy::NO_CHECK);
    #endif
}


template <typename T> //NOLINT(misc-unconventional-assign-operator)
inline STDGPU_HOST_DEVICE T //NOLINT(misc-unconventional-assign-operator)
atomic_ref<T>::operator=(const T desired)
{
    store(desired);

    return desired;
}


template <typename T>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::exchange(const T desired,
                        const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_exchange(_value, desired);

    detail::atomic_store_thread_fence(order);

    return result;
}



template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic_ref<T>::compare_exchange_weak(T& expected,
                                     const T desired,
                                     const memory_order order)
{
    return compare_exchange_strong(expected, desired, order);
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic_ref<T>::compare_exchange_strong(T& expected,
                                       const T desired,
                                       const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T old = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_compare_exchange(_value, expected, desired);
    bool changed = (old == expected);

    if (!changed)
    {
        expected = old;
    }

    detail::atomic_store_thread_fence(order);

    return changed;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_add(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_add(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_sub(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_sub(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_and(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_and(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_or(const T arg,
                        const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_or(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_xor(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_xor(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_min(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_min(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_max(const T arg,
                         const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_max(_value, arg);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same<T, unsigned int>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_inc_mod(const T arg,
                             const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_inc_mod(_value, arg - 1);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_same<T, unsigned int>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_dec_mod(const T arg,
                             const memory_order order)
{
    detail::atomic_load_thread_fence(order);

    T result = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_dec_mod(_value, arg - 1);

    detail::atomic_store_thread_fence(order);

    return result;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator++()
{
    return fetch_add(1) + 1;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator++(int)
{
    return fetch_add(1);
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator--()
{
    return fetch_sub(1) - 1;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator--(int)
{
    return fetch_sub(1);
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator+=(const T arg)
{
    return fetch_add(arg) + arg;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value || std::is_floating_point<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator-=(const T arg)
{
    return fetch_sub(arg) - arg;
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator&=(const T arg)
{
    return fetch_and(arg) & arg; // NOLINT(hicpp-signed-bitwise)
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator|=(const T arg)
{
    return fetch_or(arg) | arg; // NOLINT(hicpp-signed-bitwise)
}


template <typename T>
template <STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(std::is_integral<T>::value)>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator^=(const T arg)
{
    return fetch_xor(arg) ^ arg; // NOLINT(hicpp-signed-bitwise)
}



template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE bool
atomic_is_lock_free(const atomic<T, Allocator>* obj)
{
    return obj->is_lock_free();
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE T
atomic_load(const atomic<T, Allocator>* obj)
{
    return obj->load();
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE T
atomic_load_explicit(const atomic<T, Allocator>* obj,
                     const memory_order order)
{
    return obj->load(order);
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE void
atomic_store(atomic<T, Allocator>* obj,
             const typename atomic<T, Allocator>::value_type desired)
{
    obj->store(desired);
}


template <typename T, typename Allocator>
inline STDGPU_HOST_DEVICE void
atomic_store_explicit(atomic<T, Allocator>* obj,
                      const typename atomic<T, Allocator>::value_type desired,
                      const memory_order order)
{
    obj->store(desired, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_exchange(atomic<T, Allocator>* obj,
                const typename atomic<T, Allocator>::value_type desired)
{
    return obj->exchange(desired);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_exchange_explicit(atomic<T, Allocator>* obj,
                         const typename atomic<T, Allocator>::value_type desired,
                         const memory_order order)
{
    return obj->exchange(desired, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
atomic_compare_exchange_weak(atomic<T, Allocator>* obj,
                             typename atomic<T, Allocator>::value_type* expected,
                             const typename atomic<T, Allocator>::value_type desired)
{
    return obj->compare_exchange_weak(*expected, desired);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY bool
atomic_compare_exchange_strong(atomic<T, Allocator>* obj,
                               typename atomic<T, Allocator>::value_type* expected,
                               const typename atomic<T, Allocator>::value_type desired)
{
    return obj->compare_exchange_strong(*expected, desired);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_add(atomic<T, Allocator>* obj,
                 const typename atomic<T, Allocator>::difference_type arg)
{
    return obj->fetch_add(arg);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_add_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order)
{
    return obj->fetch_add(arg, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_sub(atomic<T, Allocator>* obj,
                 const typename atomic<T, Allocator>::difference_type arg)
{
    return obj->fetch_sub(arg);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_sub_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order)
{
    return obj->fetch_sub(arg, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_and(atomic<T, Allocator>* obj,
                 const typename atomic<T, Allocator>::difference_type arg)
{
    return obj->fetch_and(arg);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_and_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order)
{
    return obj->fetch_and(arg, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_or(atomic<T, Allocator>* obj,
                const typename atomic<T, Allocator>::difference_type arg)
{
    return obj->fetch_or(arg);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_or_explicit(atomic<T, Allocator>* obj,
                         const typename atomic<T, Allocator>::difference_type arg,
                         const memory_order order)
{
    return obj->fetch_or(arg, order);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_xor(atomic<T, Allocator>* obj,
                 const typename atomic<T, Allocator>::difference_type arg)
{
    return obj->fetch_xor(arg);
}


template <typename T, typename Allocator>
inline STDGPU_DEVICE_ONLY T
atomic_fetch_xor_explicit(atomic<T, Allocator>* obj,
                          const typename atomic<T, Allocator>::difference_type arg,
                          const memory_order order)
{
    return obj->fetch_xor(arg, order);
}

} // namespace stdgpu



#endif // STDGPU_ATOMIC_DETAIL_H
