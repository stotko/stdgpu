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

#ifndef STDGPU_CMATH_H
#define STDGPU_CMATH_H

/**
 * \addtogroup cmath cmath
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/cmath.h
 */

#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \ingroup cmath
 * \brief Computes the absolute value of the given argument
 * \param[in] arg A value
 * \return arg if arg > 0.0f, -arg otherwise
 */
constexpr STDGPU_HOST_DEVICE float
abs(const float arg);

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/cmath_detail.h>



#endif // STDGPU_CMATH_H
