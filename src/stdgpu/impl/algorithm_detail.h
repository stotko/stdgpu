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

#ifndef STDGPU_ALGORTIMH_DETAIL_H
#define STDGPU_ALGORTIMH_DETAIL_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <utility>

#include <stdgpu/contract.h>

namespace stdgpu
{

template <class T>
constexpr STDGPU_HOST_DEVICE const T&
min(const T& a, const T& b)
{
    return (b < a) ? b : a;
}

template <class T>
constexpr STDGPU_HOST_DEVICE const T&
max(const T& a, const T& b)
{
    return (a < b) ? b : a;
}

template <class T>
/*constexpr*/ STDGPU_HOST_DEVICE const T&
clamp(const T& v, const T& lower, const T& upper)
{
    STDGPU_EXPECTS(!(upper < lower));

    return v < lower ? lower : upper < v ? upper : v;
}

template <typename IndexType, typename ExecutionPolicy, typename UnaryFunction>
void
for_each_index(ExecutionPolicy&& policy, IndexType size, UnaryFunction f)
{
    thrust::for_each(std::forward<ExecutionPolicy>(policy),
                     thrust::counting_iterator<IndexType>(0),
                     thrust::counting_iterator<IndexType>(size),
                     f);
}

namespace detail
{
template <typename Iterator, typename T>
class fill_functor
{
public:
    fill_functor(Iterator begin, const T& value) // NOLINT(modernize-pass-by-value)
      : _begin(begin)
      , _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index_t i)
    {
        _begin[i] = _value;
    }

private:
    Iterator _begin;
    T _value;
};
} // namespace detail

template <typename ExecutionPolicy, typename Iterator, typename T>
void
fill(ExecutionPolicy&& policy, Iterator begin, Iterator end, const T& value)
{
    fill_n(std::forward<ExecutionPolicy>(policy), begin, static_cast<index_t>(end - begin), value);
}

template <typename ExecutionPolicy, typename Iterator, typename Size, typename T>
Iterator
fill_n(ExecutionPolicy&& policy, Iterator begin, Size n, const T& value)
{
    for_each_index(std::forward<ExecutionPolicy>(policy), n, detail::fill_functor<Iterator, T>(begin, value));
    return begin + n;
}

namespace detail
{
template <typename InputIt, typename OutputIt>
class copy_functor
{
public:
    copy_functor(InputIt begin, OutputIt output_begin) // NOLINT(modernize-pass-by-value)
      : _begin(begin)
      , _output_begin(output_begin)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index_t i)
    {
        _output_begin[i] = _begin[i];
    }

private:
    InputIt _begin;
    OutputIt _output_begin;
};
} // namespace detail

template <typename ExecutionPolicy, typename InputIt, typename OutputIt>
OutputIt
copy(ExecutionPolicy&& policy, InputIt begin, InputIt end, OutputIt output_begin)
{
    return copy_n(std::forward<ExecutionPolicy>(policy), begin, static_cast<index_t>(end - begin), output_begin);
}

template <typename ExecutionPolicy, typename InputIt, typename Size, typename OutputIt>
OutputIt
copy_n(ExecutionPolicy&& policy, InputIt begin, Size n, OutputIt output_begin)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   n,
                   detail::copy_functor<InputIt, OutputIt>(begin, output_begin));
    return output_begin + n;
}

} // namespace stdgpu

#endif // STDGPU_ALGORTIMH_DETAIL_H
