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

#ifndef TEST_MEMORY_UTILS_DETAIL_H
#define TEST_MEMORY_UTILS_DETAIL_H

namespace test_utils
{

template <typename T>
STDGPU_HOST_DEVICE
test_device_allocator<T>::test_device_allocator() noexcept
{
#if STDGPU_CODE == STDGPU_CODE_HOST || STDGPU_BACKEND == STDGPU_BACKEND_OPENMP
    get_allocator_statistics().default_constructions++;
#endif
}

template <typename T>
STDGPU_HOST_DEVICE test_device_allocator<T>::~test_device_allocator() noexcept
{
#if STDGPU_CODE == STDGPU_CODE_HOST || STDGPU_BACKEND == STDGPU_BACKEND_OPENMP
    get_allocator_statistics().destructions++;
#endif
}

template <typename T>
STDGPU_HOST_DEVICE
test_device_allocator<T>::test_device_allocator([[maybe_unused]] const test_device_allocator& other) noexcept
{
#if STDGPU_CODE == STDGPU_CODE_HOST || STDGPU_BACKEND == STDGPU_BACKEND_OPENMP
    get_allocator_statistics().copy_constructions++;
#endif
}

template <typename T>
template <typename U>
STDGPU_HOST_DEVICE
test_device_allocator<T>::test_device_allocator([[maybe_unused]] const test_device_allocator<U>& other) noexcept
{
#if STDGPU_CODE == STDGPU_CODE_HOST || STDGPU_BACKEND == STDGPU_BACKEND_OPENMP
    get_allocator_statistics().copy_constructions++;
#endif
}

template <typename T>
[[nodiscard]] T*
test_device_allocator<T>::allocate(stdgpu::index64_t n)
{
    return base_type().allocate(n);
}

template <typename T>
void
test_device_allocator<T>::deallocate(T* p, stdgpu::index64_t n)
{
    base_type().deallocate(p, n);
}

} // namespace test_utils

#endif // TEST_MEMORY_UTILS_DETAIL_H
