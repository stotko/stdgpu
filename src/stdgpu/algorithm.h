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

#ifndef STDGPU_ALGORITHM_H
#define STDGPU_ALGORITHM_H

/**
 * \addtogroup algorithm algorithm
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/algorithm.h
 */

#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \ingroup algorithm
 * \brief Computes the minimum of the given values
 * \tparam T The type of the values
 * \param[in] a A value
 * \param[in] b Another value
 * \return a if a <= b, b otherwise
 */
template <class T>
constexpr STDGPU_HOST_DEVICE const T&
min(const T& a,
    const T& b);

/**
 * \ingroup algorithm
 * \brief Computes the maximum of the given values
 * \tparam T The type of the values
 * \param[in] a A value
 * \param[in] b Another value
 * \return a if a >= b, b otherwise
 */
template <class T>
constexpr STDGPU_HOST_DEVICE const T&
max(const T& a,
    const T& b);


/**
 * \ingroup algorithm
 * \brief Clamps a value to the given range
 * \tparam T The type of the values
 * \param[in] v A value
 * \param[in] lower The lower bound
 * \param[in] upper The upper bound
 * \return lower if v < lower, upper if upper < v, v otherwise
 * \pre !(upper < lower)
 */
template <class T>
/*constexpr*/ STDGPU_HOST_DEVICE const T&
clamp(const T& v,
      const T& lower,
      const T& upper);

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/algorithm_detail.h>



#endif // STDGPU_ALGORITHM_H
