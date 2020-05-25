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

#include <stdgpu/memory.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

template <typename T>
inline atomic<T>
atomic<T>::createDeviceObject()
{
    atomic<T> result(createDeviceArray<T>(1, 0));

    return result;
}


template <typename T>
inline
atomic<T>::atomic(T* value)
    : _value(value),
      _value_ref(_value)    // re-initialize
{

}


template <typename T>
inline void
atomic<T>::destroyDeviceObject(atomic<T>& device_object)
{
    destroyDeviceArray<T>(device_object._value);
}


template <typename T>
inline
atomic<T>::atomic()
    : _value_ref(nullptr)
{

}


template <typename T>
inline STDGPU_HOST_DEVICE T
atomic<T>::load() const
{
    return _value_ref.load();
}


template <typename T>
inline STDGPU_HOST_DEVICE
atomic<T>::operator T() const
{
    return _value_ref.operator T();
}


template <typename T>
inline STDGPU_HOST_DEVICE void
atomic<T>::store(const T desired)
{
    _value_ref.store(desired);
}


template <typename T> //NOLINT(misc-unconventional-assign-operator)
inline STDGPU_HOST_DEVICE T //NOLINT(misc-unconventional-assign-operator)
atomic<T>::operator=(const T desired)
{
    return _value_ref.operator=(desired);
}


template <typename T>
inline STDGPU_DEVICE_ONLY T
atomic<T>::exchange(const T desired)
{
    return _value_ref.exchange(desired);
}



template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic<T>::compare_exchange_weak(T& expected,
                                 const T desired)
{
    return _value_ref.compare_exchange_weak(expected, desired);
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic<T>::compare_exchange_strong(T& expected,
                                   const T desired)
{
    return _value_ref.compare_exchange_strong(expected, desired);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_add(const T arg)
{
    return _value_ref.fetch_add(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_sub(const T arg)
{
    return _value_ref.fetch_sub(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_and(const T arg)
{
    return _value_ref.fetch_and(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_or(const T arg)
{
    return _value_ref.fetch_or(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_xor(const T arg)
{
    return _value_ref.fetch_xor(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_min(const T arg)
{
    return _value_ref.fetch_min(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_max(const T arg)
{
    return _value_ref.fetch_max(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_inc_mod(const T arg)
{
    return _value_ref.fetch_inc_mod(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::fetch_dec_mod(const T arg)
{
    return _value_ref.fetch_dec_mod(arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator++()
{
    return ++_value_ref;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator++(int)
{
    return _value_ref++;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator--()
{
    return --_value_ref;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator--(int)
{
    return _value_ref--;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator+=(const T arg)
{
    return _value_ref += arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator-=(const T arg)
{
    return _value_ref -= arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator&=(const T arg)
{
    return _value_ref &= arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator|=(const T arg)
{
    return _value_ref |= arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic<T>::operator^=(const T arg)
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
inline STDGPU_HOST_DEVICE T
atomic_ref<T>::load() const
{
    if (_value == nullptr)
    {
        return 0;
    }

    T local_value;
    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        local_value = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_load(_value);
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
atomic_ref<T>::store(const T desired)
{
    if (_value == nullptr)
    {
        return;
    }

    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_store(_value, desired);
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
atomic_ref<T>::exchange(const T desired)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_exchange(_value, desired);
}



template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic_ref<T>::compare_exchange_weak(T& expected,
                                     const T desired)
{
    return compare_exchange_strong(expected, desired);
}


template <typename T>
inline STDGPU_DEVICE_ONLY bool
atomic_ref<T>::compare_exchange_strong(T& expected,
                                       const T desired)
{
    T old = stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_compare_exchange(_value, expected, desired);
    bool changed = (old == expected);

    if (!changed)
    {
        expected = old;
    }

    return changed;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_add(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_add(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_sub(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_sub(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_and(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_and(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_or(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_or(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_xor(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_xor(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_min(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_min(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_max(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_max(_value, arg);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_inc_mod(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_inc_mod(_value, arg - 1);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::fetch_dec_mod(const T arg)
{
    return stdgpu::STDGPU_BACKEND_NAMESPACE::atomic_fetch_dec_mod(_value, arg - 1);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator++()
{
    return fetch_add(1) + 1;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator++(int)
{
    return fetch_add(1);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator--()
{
    return fetch_sub(1) - 1;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator--(int)
{
    return fetch_sub(1);
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator+=(const T arg)
{
    return fetch_add(arg) + arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator-=(const T arg)
{
    return fetch_sub(arg) - arg;
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator&=(const T arg)
{
    return fetch_and(arg) & arg; // NOLINT(hicpp-signed-bitwise)
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator|=(const T arg)
{
    return fetch_or(arg) | arg; // NOLINT(hicpp-signed-bitwise)
}


template <typename T>
template <typename U, typename>
inline STDGPU_DEVICE_ONLY T
atomic_ref<T>::operator^=(const T arg)
{
    return fetch_xor(arg) ^ arg; // NOLINT(hicpp-signed-bitwise)
}

} // namespace stdgpu



#endif // STDGPU_ATOMIC_DETAIL_H
