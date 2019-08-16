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

#include <stdgpu/contract.h>



namespace stdgpu
{

template <class T>
constexpr STDGPU_HOST_DEVICE const T&
min(const T& a,
    const T& b)
{
    return (b < a) ? b : a;
}


template <class T>
constexpr STDGPU_HOST_DEVICE const T&
max(const T& a,
    const T& b)
{
    return (a < b) ? b : a;
}


template <class T>
/*constexpr*/ STDGPU_HOST_DEVICE const T&
clamp(const T& v,
      const T& lower,
      const T& upper)
{
    STDGPU_EXPECTS(!(upper < lower));

    return v < lower ? lower : upper < v ? upper : v;
}

} // namespace stdgpu



#endif // STDGPU_ALGORTIMH_DETAIL_H
