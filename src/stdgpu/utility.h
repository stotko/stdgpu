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

#ifndef STDGPU_UTILITY_H
#define STDGPU_UTILITY_H

/**
 * \addtogroup utility utility
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/utility.h
 */

#include <type_traits>

#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \ingroup utility
 * \brief Forwards a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The forwarded value
 */
template< class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>& t) noexcept;


/**
 * \ingroup utility
 * \brief Forwards a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The forwarded value
 */
template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>&& t) noexcept;


/**
 * \ingroup utility
 * \brief Moves a value
 * \tparam T The type of the value
 * \param[in] t A value
 * \return The moved value
 */
template<class T>
constexpr STDGPU_HOST_DEVICE std::remove_reference_t<T>&&
move(T&& t) noexcept;

} // namespace stdgpu



/**
 * @}
 */



#include <stdgpu/impl/utility_detail.h>



#endif // STDGPU_UTILITY_H
