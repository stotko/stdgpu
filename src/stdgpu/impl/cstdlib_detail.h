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

#ifndef STDGPU_CSTDLIB_DETAIL_H
#define STDGPU_CSTDLIB_DETAIL_H

#include <stdgpu/bit.h>
#include <stdgpu/contract.h>



namespace stdgpu
{

inline STDGPU_HOST_DEVICE sizediv_t
sizedivPow2(const std::size_t x,
            const std::size_t y)
{
    STDGPU_EXPECTS(y > 0);

    sizediv_t result;
    result.quot = x / y;
    result.rem  = bit_mod(x, y);

    STDGPU_ENSURES(result.quot * y + result.rem == x);

    return result;
}

} // namespace stdgpu



#endif // STDGPU_CSTDLIB_DETAIL_H
