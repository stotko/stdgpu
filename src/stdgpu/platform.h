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
 * \file stdgpu/platform.h
 */

#include <stdgpu/config.h>

//! @cond Doxygen_Suppress
#define STDGPU_BACKEND_PLATFORM_HEADER <stdgpu/STDGPU_BACKEND_DIRECTORY/platform.h> // NOLINT(bugprone-macro-parentheses,misc-macro-parentheses)
#include STDGPU_BACKEND_PLATFORM_HEADER
#undef STDGPU_BACKEND_PLATFORM_HEADER
//! @endcond

// NOTE: For backwards compatibility only
#include <stdgpu/compiler.h>



namespace stdgpu
{

/**
 * \def STDGPU_HAS_CXX_17
 * \hideinitializer
 * \brief Indicator of C++17 availability
 */
#if defined(__cplusplus) && __cplusplus >= 201703L
    #define STDGPU_HAS_CXX_17 1
#else
    #define STDGPU_HAS_CXX_17 0
#endif



/**
 * \hideinitializer
 * \brief Backend: CUDA
 */
#define STDGPU_BACKEND_CUDA   100
/**
 * \hideinitializer
 * \brief Backend: OpenMP
 */
#define STDGPU_BACKEND_OPENMP 101
/**
 * \hideinitializer
 * \brief Backend: ROCm
 */
#define STDGPU_BACKEND_ROCM   102


// STDGPU_BACKEND is defined in stdgpu/config.h


namespace detail
{

//! @cond Doxygen_Suppress
#define STDGPU_DETAIL_CAT2_DIRECT(A, B) A ## B
#define STDGPU_DETAIL_CAT2(A, B) STDGPU_DETAIL_CAT2_DIRECT(A, B)
#define STDGPU_DETAIL_CAT3(A, B, C) STDGPU_DETAIL_CAT2(A, STDGPU_DETAIL_CAT2(B, C))
//! @endcond

} // namespace detail


/**
 * \def STDGPU_HOST_DEVICE
 * \hideinitializer
 * \brief Platform-independent host device function annotation
 */
#define STDGPU_HOST_DEVICE STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _HOST_DEVICE)


/**
 * \def STDGPU_DEVICE_ONLY
 * \hideinitializer
 * \brief Platform-independent device function annotation
 */
#define STDGPU_DEVICE_ONLY STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _DEVICE_ONLY)


/**
 * \def STDGPU_CONSTANT
 * \hideinitializer
 * \brief Platform-independent constant variable annotation
 */
#define STDGPU_CONSTANT STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _CONSTANT)


/**
 * \hideinitializer
 * \brief Code path: Host
 */
#define STDGPU_CODE_HOST   1000
/**
 * \hideinitializer
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
 * \def STDGPU_CODE
 * \hideinitializer
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
#define STDGPU_DETAIL_IS_DEVICE_COMPILED STDGPU_DETAIL_CAT3(STDGPU_, STDGPU_BACKEND_MACRO_NAMESPACE, _IS_DEVICE_COMPILED)
//! @endcond

} // namespace detail


} // namespace stdgpu



#endif // STDGPU_PLATFORM_H
