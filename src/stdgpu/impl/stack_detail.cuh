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

#ifndef STDGPU_STACK_DETAIL_H
#define STDGPU_STACK_DETAIL_H

namespace stdgpu
{

template <typename T, typename ContainerT>
stack<T, ContainerT>
stack<T, ContainerT>::createDeviceObject(const index_t& size)
{
    STDGPU_EXPECTS(size > 0);

    stack<T, ContainerT> result;
    result._c = ContainerT::createDeviceObject(size);

    return result;
}

template <typename T, typename ContainerT>
void
stack<T, ContainerT>::destroyDeviceObject(stack<T, ContainerT>& device_object)
{
    ContainerT::destroyDeviceObject(device_object._c);
}

template <typename T, typename ContainerT>
inline STDGPU_DEVICE_ONLY bool
stack<T, ContainerT>::push(const T& element)
{
    return _c.push_back(element);
}

template <typename T, typename ContainerT>
inline STDGPU_DEVICE_ONLY pair<T, bool>
stack<T, ContainerT>::pop()
{
    return _c.pop_back();
}

template <typename T, typename ContainerT>
inline STDGPU_HOST_DEVICE bool
stack<T, ContainerT>::empty() const
{
    return _c.empty();
}

template <typename T, typename ContainerT>
inline STDGPU_HOST_DEVICE bool
stack<T, ContainerT>::full() const
{
    return _c.full();
}

template <typename T, typename ContainerT>
inline STDGPU_HOST_DEVICE index_t
stack<T, ContainerT>::size() const
{
    return _c.size();
}

template <typename T, typename ContainerT>
inline STDGPU_HOST_DEVICE index_t
stack<T, ContainerT>::capacity() const
{
    return _c.capacity();
}

template <typename T, typename ContainerT>
inline bool
stack<T, ContainerT>::valid() const
{
    return _c.valid();
}

} // namespace stdgpu

#endif // STDGPU_STACK_DETAIL_H
