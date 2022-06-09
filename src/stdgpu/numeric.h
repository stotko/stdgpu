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

#ifndef STDGPU_NUMERIC_H
#define STDGPU_NUMERIC_H

/**
 * \addtogroup numeric numeric
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/numeric.h
 */

namespace stdgpu
{

/**
 * \ingroup numeric
 * \brief Writes ascending values {values + i} to the i-th position of the given range
 * \tparam ExecutionPolicy The type of the execution policy
 * \tparam Iterator The type of the iterators
 * \tparam T The type of the values
 * \param[in] policy The execution policy, e.g. host or device
 * \param[in] begin The iterator pointing to the first element
 * \param[in] end The iterator pointing past to the last element
 * \param[in] value The starting value that will be incremented
 */
template <typename ExecutionPolicy, typename Iterator, typename T>
void
iota(ExecutionPolicy&& policy, Iterator begin, Iterator end, T value);

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/numeric_detail.h>

#endif // STDGPU_NUMERIC_H
