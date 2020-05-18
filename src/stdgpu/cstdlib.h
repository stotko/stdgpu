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

#ifndef STDGPU_CSTDLIB_H
#define STDGPU_CSTDLIB_H

/**
 * \addtogroup cstdlib cstdlib
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/cstdlib.h
 */

#include <cstddef>

#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \brief A struct to capture the result of sizedivPow2
 * \deprecated Use x / y and bit_mod(x, y) directly
 */
struct /*[[deprecated(" Use x / y and bit_mod(x, y) directly")]]*/ sizediv_t
{
    std::size_t quot = {};  /**< The quotient */
    std::size_t rem = {};   /**< The remainder */
};

/**
 * \ingroup cstdlib
 * \brief Computes x/y and x%y where y = 2^n
 * \param[in] x A number
 * \param[in] y The divider with y = 2^n
 * \return The resulting quotient and remainder
 * \pre y > 0
 * \pre has_single_bit(y)
 * \post result.quot * y + result.rem == x
 * \deprecated Use x / y and bit_mod(x, y) directly
 */
[[deprecated(" Use x / y and bit_mod(x, y) directly")]]
STDGPU_HOST_DEVICE sizediv_t
sizedivPow2(const std::size_t x,
            const std::size_t y);

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/cstdlib_detail.h>



#endif // STDGPU_CSTDLIB_H
