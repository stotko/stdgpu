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

#ifndef STDGPU_ATTRIBUTE_H
#define STDGPU_ATTRIBUTE_H

/**
 * \addtogroup attribute attribute
 * \ingroup system
 * @{
 */

/**
 * \file stdgpu/attribute.h
 */

#include <stdgpu/compiler.h>



namespace stdgpu
{

//! @cond Doxygen_Suppress
#ifdef __has_cpp_attribute
    #define STDGPU_HAS_CPP_ATTRIBUTE(name) __has_cpp_attribute(name)
#else
    #define STDGPU_HAS_CPP_ATTRIBUTE(name) 0
#endif
//! @endcond


/**
 * \ingroup attribute
 * \def STDGPU_MAYBE_UNUSED
 * \brief Suppresses compiler warnings caused by variables that are or might be unused
 */
#if STDGPU_HAS_CPP_ATTRIBUTE(maybe_unused) && (STDGPU_HAS_CXX_17 || STDGPU_HOST_COMPILER != STDGPU_HOST_COMPILER_CLANG) // Clang outputs a warning if C++17 is not present
    #define STDGPU_MAYBE_UNUSED [[maybe_unused]]
#elif STDGPU_HAS_CPP_ATTRIBUTE(gnu::unused)
    #define STDGPU_MAYBE_UNUSED [[gnu::unused]]
#elif STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
    #define STDGPU_MAYBE_UNUSED __pragma(warning(suppress: 4100 4101))
#else
    #define STDGPU_MAYBE_UNUSED
#endif


/**
 * \ingroup attribute
 * \def STDGPU_FALLTHROUGH
 * \brief Suppresses compiler warnings caused by implicit fallthrough
 */
#if STDGPU_HAS_CPP_ATTRIBUTE(fallthrough) && (STDGPU_HAS_CXX_17 || STDGPU_HOST_COMPILER != STDGPU_HOST_COMPILER_CLANG)  // Clang outputs a warning if C++17 is not present
    #define STDGPU_FALLTHROUGH [[fallthrough]]
#elif STDGPU_HAS_CPP_ATTRIBUTE(gnu::fallthrough)
    #define STDGPU_FALLTHROUGH [[gnu::fallthrough]]
#elif STDGPU_HAS_CPP_ATTRIBUTE(clang::fallthrough)
    #define STDGPU_FALLTHROUGH [[clang::fallthrough]]
#else
    #define STDGPU_FALLTHROUGH
#endif


/**
 * \ingroup attribute
 * \def STDGPU_NODISCARD
 * \brief Encourages compiler warnings or errors if the function return value is unused
 */
#if STDGPU_HAS_CPP_ATTRIBUTE(nodiscard) && (STDGPU_HAS_CXX_17 || STDGPU_HOST_COMPILER != STDGPU_HOST_COMPILER_CLANG)    // Clang outputs a warning if C++17 is not present
    #define STDGPU_NODISCARD [[nodiscard]]
#elif STDGPU_HAS_CPP_ATTRIBUTE(gnu::warn_unused_result)
    #define STDGPU_NODISCARD [[gnu::warn_unused_result]]
#elif STDGPU_HAS_CPP_ATTRIBUTE(clang::warn_unused_result)
    #define STDGPU_NODISCARD [[clang::warn_unused_result]]
#elif STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
    #define STDGPU_NODISCARD _Check_return_
#else
    #define STDGPU_NODISCARD
#endif

} // namespace stdgpu



/**
 * @}
 */



#endif // STDGPU_ATTRIBUTE_H
