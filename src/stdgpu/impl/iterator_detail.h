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

#ifndef STDGPU_ITERATORDETAIL_H
#define STDGPU_ITERATORDETAIL_H

#include <cstdio>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <type_traits>

namespace stdgpu
{

template <typename T>
STDGPU_HOST_DEVICE device_ptr<T>
make_device(T* device_array)
{
    return device_ptr<T>(device_array);
}

template <typename T>
STDGPU_HOST_DEVICE host_ptr<T>
make_host(T* host_array)
{
    return host_ptr<T>(host_array);
}

template <typename T>
index64_t
size(T* array)
{
    index64_t array_size_bytes = size<void>(static_cast<void*>(const_cast<std::remove_cv_t<T>*>(array)));

    if (array_size_bytes % static_cast<index64_t>(sizeof(T)) != 0) // NOLINT(bugprone-sizeof-expression)
    {
        printf("stdgpu::size : Array type not matching the memory alignment. Returning 0 ...\n");
        return 0;
    }

    return array_size_bytes / static_cast<index64_t>(sizeof(T)); // NOLINT(bugprone-sizeof-expression)
}

template <typename T>
host_ptr<T>
host_begin(T* host_array)
{
    return make_host(host_array);
}

template <typename T>
host_ptr<T>
host_end(T* host_array)
{
    return make_host(host_array + size(host_array));
}

template <typename T>
device_ptr<T>
device_begin(T* device_array)
{
    return make_device(device_array);
}

template <typename T>
device_ptr<T>
device_end(T* device_array)
{
    return make_device(device_array + size(device_array));
}

template <typename T>
host_ptr<const T>
host_begin(const T* host_array)
{
    return host_cbegin(host_array);
}

template <typename T>
host_ptr<const T>
host_end(const T* host_array)
{
    return host_cend(host_array);
}

template <typename T>
device_ptr<const T>
device_begin(const T* device_array)
{
    return device_cbegin(device_array);
}

template <typename T>
device_ptr<const T>
device_end(const T* device_array)
{
    return device_cend(device_array);
}

template <typename T>
host_ptr<const T>
host_cbegin(const T* host_array)
{
    return make_host(host_array);
}

template <typename T>
host_ptr<const T>
host_cend(const T* host_array)
{
    return make_host(host_array + size(host_array));
}

template <typename T>
device_ptr<const T>
device_cbegin(const T* device_array)
{
    return make_device(device_array);
}

template <typename T>
device_ptr<const T>
device_cend(const T* device_array)
{
    return make_device(device_array + size(device_array));
}

template <typename C>
auto
host_begin(C& host_container) -> decltype(host_container.host_begin())
{
    return host_container.host_begin();
}

template <typename C>
auto
host_end(C& host_container) -> decltype(host_container.host_end())
{
    return host_container.host_end();
}

template <typename C>
auto
device_begin(C& device_container) -> decltype(device_container.device_begin())
{
    return device_container.device_begin();
}

template <typename C>
auto
device_end(C& device_container) -> decltype(device_container.device_end())
{
    return device_container.device_end();
}

template <typename C>
auto
host_begin(const C& host_container) -> decltype(host_container.host_begin())
{
    return host_container.host_begin();
}

template <typename C>
auto
host_end(const C& host_container) -> decltype(host_container.host_end())
{
    return host_container.host_end();
}

template <typename C>
auto
device_begin(const C& device_container) -> decltype(device_container.device_begin())
{
    return device_container.device_begin();
}

template <typename C>
auto
device_end(const C& device_container) -> decltype(device_container.device_end())
{
    return device_container.device_end();
}

template <typename C>
auto
host_cbegin(const C& host_container) -> decltype(host_begin(host_container))
{
    return host_begin(host_container);
}

template <typename C>
auto
host_cend(const C& host_container) -> decltype(host_end(host_container))
{
    return host_end(host_container);
}

template <typename C>
auto
device_cbegin(const C& device_container) -> decltype(device_begin(device_container))
{
    return device_begin(device_container);
}

template <typename C>
auto
device_cend(const C& device_container) -> decltype(device_end(device_container))
{
    return device_end(device_container);
}

namespace detail
{

template <typename Container>
class back_insert_iterator_proxy
{
public:
    STDGPU_HOST_DEVICE
    explicit back_insert_iterator_proxy(const Container& c)
      : _c(c)
    {
    }

    STDGPU_HOST_DEVICE back_insert_iterator_proxy&
    operator=(const typename Container::value_type& value)
    {
        _c.push_back(value);
        return *this;
    }

private:
    Container _c;
};

template <typename Container>
struct back_insert_iterator_base
{
    using type = thrust::iterator_adaptor<back_insert_iterator<Container>,
                                          thrust::discard_iterator<>,
                                          thrust::use_default,
                                          thrust::use_default,
                                          thrust::use_default,
                                          back_insert_iterator_proxy<Container>>;
};

template <typename Container>
class front_insert_iterator_proxy
{
public:
    STDGPU_HOST_DEVICE
    explicit front_insert_iterator_proxy(const Container& c)
      : _c(c)
    {
    }

    STDGPU_HOST_DEVICE front_insert_iterator_proxy&
    operator=(const typename Container::value_type& value)
    {
        _c.push_front(value);
        return *this;
    }

private:
    Container _c;
};

template <typename Container>
struct front_insert_iterator_base
{
    using type = thrust::iterator_adaptor<front_insert_iterator<Container>,
                                          thrust::discard_iterator<>,
                                          thrust::use_default,
                                          thrust::use_default,
                                          thrust::use_default,
                                          front_insert_iterator_proxy<Container>>;
};

template <typename Container>
class insert_iterator_proxy
{
public:
    STDGPU_HOST_DEVICE
    explicit insert_iterator_proxy(const Container& c)
      : _c(c)
    {
    }

    STDGPU_HOST_DEVICE insert_iterator_proxy&
    operator=(const typename Container::value_type& value)
    {
        _c.insert(value);
        return *this;
    }

private:
    Container _c;
};

template <typename Container>
struct insert_iterator_base
{
    using type = thrust::iterator_adaptor<insert_iterator<Container>,
                                          thrust::discard_iterator<>,
                                          thrust::use_default,
                                          thrust::use_default,
                                          thrust::use_default,
                                          insert_iterator_proxy<Container>>;
};

} // namespace detail

template <typename Container>
STDGPU_HOST_DEVICE
back_insert_iterator<Container>::back_insert_iterator(Container& c)
  : _c(c)
{
}

template <typename Container>
STDGPU_HOST_DEVICE typename back_insert_iterator<Container>::super_t::reference
back_insert_iterator<Container>::dereference() const
{
    return detail::back_insert_iterator_proxy<Container>(_c);
}

template <typename Container>
STDGPU_HOST_DEVICE back_insert_iterator<Container>
back_inserter(Container& c)
{
    return back_insert_iterator<Container>(c);
}

template <typename Container>
STDGPU_HOST_DEVICE
front_insert_iterator<Container>::front_insert_iterator(Container& c)
  : _c(c)
{
}

template <typename Container>
STDGPU_HOST_DEVICE typename front_insert_iterator<Container>::super_t::reference
front_insert_iterator<Container>::dereference() const
{
    return detail::front_insert_iterator_proxy<Container>(_c);
}

template <typename Container>
STDGPU_HOST_DEVICE front_insert_iterator<Container>
front_inserter(Container& c)
{
    return front_insert_iterator<Container>(c);
}

template <typename Container>
STDGPU_HOST_DEVICE
insert_iterator<Container>::insert_iterator(Container& c)
  : _c(c)
{
}

template <typename Container>
STDGPU_HOST_DEVICE typename insert_iterator<Container>::super_t::reference
insert_iterator<Container>::dereference() const
{
    return detail::insert_iterator_proxy<Container>(_c);
}

template <typename Container>
STDGPU_HOST_DEVICE insert_iterator<Container>
inserter(Container& c)
{
    return insert_iterator<Container>(c);
}

} // namespace stdgpu

namespace thrust::detail
{

template <typename Container>
struct is_proxy_reference<stdgpu::detail::back_insert_iterator_proxy<Container>> : public thrust::detail::true_type
{
};

template <typename Container>
struct is_proxy_reference<stdgpu::detail::front_insert_iterator_proxy<Container>> : public thrust::detail::true_type
{
};

template <typename Container>
struct is_proxy_reference<stdgpu::detail::insert_iterator_proxy<Container>> : public thrust::detail::true_type
{
};

} // namespace thrust::detail

#endif // STDGPU_ITERATORDETAIL_H
