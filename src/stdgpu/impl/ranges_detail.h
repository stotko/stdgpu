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
device_range<T>::device_range(T* p, index64_t n)
  : _begin(p)
  , _end(p + n)
{
}

template <typename T>
STDGPU_HOST_DEVICE
device_range<T>::device_range(typename device_range<T>::iterator begin, index64_t n)
  : _begin(begin)
  , _end(begin + n)
{
}

template <typename T>
STDGPU_HOST_DEVICE
device_range<T>::device_range(typename device_range<T>::iterator begin, typename device_range<T>::iterator end)
  : _begin(begin)
  , _end(end)
{
}

template <typename T>
STDGPU_HOST_DEVICE typename device_range<T>::iterator
device_range<T>::begin() const noexcept
{
    return _begin;
}

template <typename T>
STDGPU_HOST_DEVICE typename device_range<T>::iterator
device_range<T>::end() const noexcept
{
    return _end;
}

template <typename T>
STDGPU_HOST_DEVICE index64_t
device_range<T>::size() const
{
    return end() - begin();
}

template <typename T>
STDGPU_HOST_DEVICE bool
device_range<T>::empty() const
{
    return size() == 0;
}

template <typename T>
host_range<T>::host_range(T* p)
  : host_range(p, stdgpu::size(p))
{
}

template <typename T>
STDGPU_HOST_DEVICE
host_range<T>::host_range(T* p, index64_t n)
  : _begin(p)
  , _end(p + n)
{
}

template <typename T>
STDGPU_HOST_DEVICE
host_range<T>::host_range(typename host_range<T>::iterator begin, index64_t n)
  : _begin(begin)
  , _end(begin + n)
{
}

template <typename T>
STDGPU_HOST_DEVICE
host_range<T>::host_range(typename host_range<T>::iterator begin, typename host_range<T>::iterator end)
  : _begin(begin)
  , _end(end)
{
}

template <typename T>
STDGPU_HOST_DEVICE typename host_range<T>::iterator
host_range<T>::begin() const noexcept
{
    return _begin;
}

template <typename T>
STDGPU_HOST_DEVICE typename host_range<T>::iterator
host_range<T>::end() const noexcept
{
    return _end;
}

template <typename T>
STDGPU_HOST_DEVICE index64_t
host_range<T>::size() const
{
    return end() - begin();
}

template <typename T>
STDGPU_HOST_DEVICE bool
host_range<T>::empty() const
{
    return size() == 0;
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE
transform_range<R, UnaryFunction>::transform_range(R r)
  : transform_range(r, UnaryFunction())
{
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE
transform_range<R, UnaryFunction>::transform_range(R r, UnaryFunction f)
  : _begin(r.begin(), f)
  , _end(r.end(), f)
{
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE typename transform_range<R, UnaryFunction>::iterator
transform_range<R, UnaryFunction>::begin() const noexcept
{
    return _begin;
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE typename transform_range<R, UnaryFunction>::iterator
transform_range<R, UnaryFunction>::end() const noexcept
{
    return _end;
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE index64_t
transform_range<R, UnaryFunction>::size() const
{
    return end() - begin();
}

template <typename R, typename UnaryFunction>
STDGPU_HOST_DEVICE bool
transform_range<R, UnaryFunction>::empty() const
{
    return size() == 0;
}

namespace detail
{

template <typename T>
class select
{
public:
    select() = default;

    // NOTE
    // Implicit conversion required for {host,device}_indexed_range:
    // Usage via constructor with arguments {host,device}_range<index_t>, T*
    STDGPU_HOST_DEVICE
    select(T* values) // NOLINT(hicpp-explicit-conversions)
      : _values(values)
    {
    }

    STDGPU_HOST_DEVICE T
    operator()(const index_t i) const
    {
        return _values[i];
    }

private:
    T* _values = nullptr;
};

} // namespace detail

} // namespace stdgpu

#endif // STDGPU_RANGES_DETAIL_H
