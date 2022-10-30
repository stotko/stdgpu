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

#ifndef STDGPU_PLATFORM_H
#define STDGPU_PLATFORM_H

/**
 * \addtogroup platform platform
 * \ingroup system
 * @{
 */

/**
 * \file stdgpu/platform.h
 */

#include <stdgpu/config.h>

//! @cond Doxygen_Suppress
/* clang-format off */
#define STDGPU_DETAIL_BACKEND_HEADER(header_file)                                                                      \
    <stdgpu/STDGPU_BACKEND_DIRECTORY/header_file> // NOLINT(bugprone-macro-parentheses,misc-macro-parentheses)
/* clang-format on */
//! @endcond

#include STDGPU_DETAIL_BACKEND_HEADER(platform.h)

#include <stdgpu/compiler.h> // NOTE: For backwards compatibility only
#include <stdgpu/impl/preprocessor.h>

namespace stdgpu
{

/**
 * \ingroup platform
 * \brief Backend: CUDA
 */
#define STDGPU_BACKEND_CUDA 100
/**
 * \ingroup platform
 * \brief Backend: OpenMP
 */
#define STDGPU_BACKEND_OPENMP 101
/**
 * \ingroup platform
 * \brief Backend: HIP
 */
#define STDGPU_BACKEND_HIP 102

/**
 * \ingroup platform
 * \def STDGPU_BACKEND
 * \brief Selected backend
 */
// Workaround: Provide a define only for the purpose of creating the documentation
#ifdef STDGPU_RUN_DOXYGEN
    #define STDGPU_BACKEND
#endif
// STDGPU_BACKEND is defined in stdgpu/config.h

/**
 * \ingroup platform
 * \def STDGPU_HOST_DEVICE
 * \brief Platform-independent host device function annotation
 */
#define STDGPU_HOST_DEVICE STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _HOST_DEVICE)

/**
 * \ingroup platform
 * \def STDGPU_DEVICE_ONLY
 * \brief Platform-independent device function annotation
 */
#define STDGPU_DEVICE_ONLY STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _DEVICE_ONLY)

/**
 * \ingroup platform
 * \def STDGPU_CONSTANT
 * \brief Platform-independent constant variable annotation
 */
#define STDGPU_CONSTANT STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _CONSTANT)

/**
 * \ingroup platform
 * \brief Code path: Host
 */
#define STDGPU_CODE_HOST 1000
/**
 * \ingroup platform
 * \brief Code path: Device
 */
#define STDGPU_CODE_DEVICE 1001

namespace detail
{

//! @cond Doxygen_Suppress
#define STDGPU_DETAIL_IS_DEVICE_CODE STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _IS_DEVICE_CODE)
//! @endcond

} // namespace detail

/**
 * \ingroup platform
 * \def STDGPU_CODE
 * \brief The code path
 */
#if STDGPU_DETAIL_IS_DEVICE_CODE
    #define STDGPU_CODE STDGPU_CODE_DEVICE
#else
    #define STDGPU_CODE STDGPU_CODE_HOST
#endif

namespace detail
{

//! @cond Doxygen_Suppress
#define STDGPU_DETAIL_IS_DEVICE_COMPILED                                                                               \
    STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _IS_DEVICE_COMPILED)
//! @endcond

} // namespace detail

} // namespace stdgpu

/**
 * @}
 */

#endif // STDGPU_PLATFORM_H
