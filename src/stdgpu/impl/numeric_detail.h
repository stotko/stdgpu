/*
 *  Copyright 2022 Patrick Stotko
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

#ifndef STDGPU_NUMERIC_DETAIL_H
#define STDGPU_NUMERIC_DETAIL_H

#include <stdgpu/algorithm.h>

namespace stdgpu
{

namespace detail
{
template <typename Iterator, typename T>
class iota_functor
{
public:
    iota_functor(Iterator begin, T value)
      : _begin(begin)
      , _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index_t i)
    {
        _begin[i] = _value + static_cast<T>(i);
    }

private:
    Iterator _begin;
    T _value;
};
} // namespace detail

template <typename ExecutionPolicy, typename Iterator, typename T>
void
iota(ExecutionPolicy&& policy, Iterator begin, Iterator end, T value)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   static_cast<index_t>(end - begin),
                   detail::iota_functor<Iterator, T>(begin, value));
}

} // namespace stdgpu

#endif // STDGPU_NUMERIC_DETAIL_H
