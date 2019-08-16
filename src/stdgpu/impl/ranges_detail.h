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

#ifndef STDGPU_RANGES_DETAIL_H
#define STDGPU_RANGES_DETAIL_H



namespace stdgpu
{

template <typename T>
device_range<T>::device_range(T* p)
    : device_range(p, stdgpu::size(p))
{

}


template <typename T>
STDGPU_HOST_DEVICE
device_range<T>::device_range(T* p,
                              index_t n)
    : _begin(p),
      _end(p + n)
{

}


template <typename T>
STDGPU_HOST_DEVICE typename device_range<T>::iterator
device_range<T>::begin()
{
    return _begin;
}


template <typename T>
STDGPU_HOST_DEVICE typename device_range<T>::iterator
device_range<T>::end()
{
    return _end;
}


template <typename T>
STDGPU_HOST_DEVICE
host_range<T>::host_range(T* p,
                          index_t n)
    : _begin(p),
      _end(p + n)
{

}


template <typename T>
host_range<T>::host_range(T* p)
    : host_range(p, stdgpu::size(p))
{

}


template <typename T>
STDGPU_HOST_DEVICE typename host_range<T>::iterator
host_range<T>::begin()
{
    return _begin;
}


template <typename T>
STDGPU_HOST_DEVICE typename host_range<T>::iterator
host_range<T>::end()
{
    return _end;
}


template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE
transform_range<R, UnaryFunction>::transform_range(R r,
                                                   UnaryFunction f)
    : _begin(r.begin(), f),
      _end(r.end(), f)
{

}


template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE typename transform_range<R, UnaryFunction>::iterator
transform_range<R, UnaryFunction>::begin()
{
    return _begin;
}


template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE typename transform_range<R, UnaryFunction>::iterator
transform_range<R, UnaryFunction>::end()
{
    return _end;
}


namespace detail
{

template <typename T>
struct select
{
    STDGPU_HOST_DEVICE
    select(T* values)
        : _values(values)
    {

    }

    STDGPU_HOST_DEVICE T
    operator()(const index_t i) const
    {
        return _values[i];
    }

    T* _values;
};

} // namespace detail

} // namespace stdgpu



#endif // STDGPU_RANGES_DETAIL_H
