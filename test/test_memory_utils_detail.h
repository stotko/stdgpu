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
test_device_allocator<T>::test_device_allocator()
{
    get_allocator_statistics().default_constructions++;
}


template <typename T>
test_device_allocator<T>::~test_device_allocator()
{
    get_allocator_statistics().destructions++;
}


template <typename T>
test_device_allocator<T>::test_device_allocator(const test_device_allocator& other)
    : _base_allocator(other._base_allocator)
{
    get_allocator_statistics().copy_constructions++;
}


template <typename T>
template <typename U>
test_device_allocator<T>::test_device_allocator(const test_device_allocator<U>& other)
    : _base_allocator(other._base_allocator)
{
    get_allocator_statistics().copy_constructions++;
}


template <typename T>
STDGPU_NODISCARD T*
test_device_allocator<T>::allocate(stdgpu::index64_t n)
{
    return _base_allocator.allocate(n);
}


template <typename T>
void
test_device_allocator<T>::deallocate(T* p,
                                     stdgpu::index64_t n)
{
    _base_allocator.deallocate(p, n);
}

}



#endif // TEST_MEMORY_UTILS_DETAIL_H
