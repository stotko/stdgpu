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

#ifndef STDGPU_CSTDDEF_H
#define STDGPU_CSTDDEF_H

/**
 * \addtogroup cstddef cstddef
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/cstddef.h
 */

#include <cinttypes>
#include <cstddef>
#include <cstdint>

#include <stdgpu/compiler.h>
#include <stdgpu/config.h>

namespace stdgpu
{

/**
 * \ingroup cstddef
 * \brief std::int_least32_t
 */
using index32_t = std::int_least32_t;

/**
 * \ingroup cstddef
 * \brief std::ptrdiff_t
 */
using index64_t = std::ptrdiff_t;

/**
 * \ingroup cstddef
 * \typedef index_t
 * \brief index32_t if STDGPU_USE_32_BIT_INDEX is set, index64_t otherwise
 */
#if STDGPU_USE_32_BIT_INDEX
using index_t = index32_t;
#else
using index_t = index64_t;
#endif

/**
 * \ingroup cstddef
 * \brief Format constant for index32_t
 */
#define STDGPU_PRIINDEX32 PRIdLEAST32

/**
 * \ingroup cstddef
 * \brief Format constant for index64_t
 */
#define STDGPU_PRIINDEX64 "td"

/**
 * \ingroup cstddef
 * \def STDGPU_PRIINDEX
 * \brief STDGPU_PRIINDEX32 if STDGPU_USE_32_BIT_INDEX is set, STDGPU_PRIINDEX32 otherwise
 */
#if STDGPU_USE_32_BIT_INDEX
    #define STDGPU_PRIINDEX STDGPU_PRIINDEX32
#else
    #define STDGPU_PRIINDEX STDGPU_PRIINDEX64
#endif

/**
 * \ingroup cstddef
 * \def STDGPU_FUNC
 * \brief A macro for getting the name of the function where this macro is expanded
 */
#if STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_GCC || STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_CLANG
    #define STDGPU_FUNC __PRETTY_FUNCTION__
#elif STDGPU_HOST_COMPILER == STDGPU_HOST_COMPILER_MSVC
    #define STDGPU_FUNC __FUNCSIG__
#else
    #define STDGPU_FUNC __func__
#endif

} // namespace stdgpu

/**
 * @}
 */

#endif // STDGPU_CSTDDEF_H
