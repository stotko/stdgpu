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

#ifndef STDGPU_CONTRACT_H
#define STDGPU_CONTRACT_H

/**
 * \addtogroup contract contract
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/contract.h
 */

#include <cassert>
#include <cstdio>
#include <exception>

#include <stdgpu/compiler.h>
#include <stdgpu/config.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>

//! @cond Doxygen_Suppress
// NOTE: CUDA-Clang uses merged parsing and needs a device version of std::terminate
#if STDGPU_DEVICE_COMPILER == STDGPU_DEVICE_COMPILER_CUDACLANG
namespace std
{
STDGPU_CUDA_DEVICE_ONLY void
terminate()
{
    // Dummy function for parsing only
}
} // namespace std
#endif
//! @endcond

namespace stdgpu
{

/**
 * \ingroup contract
 * \def STDGPU_EXPECTS(condition)
 * \brief A macro to define pre-conditions for both host and device
 */
/**
 * \ingroup contract
 * \def STDGPU_ENSURES(condition)
 * \brief A macro to define post-conditions for both host and device
 */
/**
 * \ingroup contract
 * \def STDGPU_ASSERT(condition)
 * \brief A macro to define in-body conditions for both host and device
 */

//! @cond Doxygen_Suppress
#define STDGPU_DETAIL_EMPTY_STATEMENT (void)0
//! @endcond

#if STDGPU_ENABLE_CONTRACT_CHECKS

    #define STDGPU_DETAIL_HOST_CHECK(type, ...)                                                                        \
        if (!(__VA_ARGS__))                                                                                            \
        {                                                                                                              \
            printf("stdgpu : " type " failure :\n"                                                                     \
                   "  File      : %s:%d\n"                                                                             \
                   "  Function  : %s\n"                                                                                \
                   "  Condition : %s\n",                                                                               \
                   __FILE__,                                                                                           \
                   __LINE__,                                                                                           \
                   static_cast<const char*>(STDGPU_FUNC),                                                              \
                   #__VA_ARGS__);                                                                                      \
            std::terminate();                                                                                          \
        }                                                                                                              \
        STDGPU_DETAIL_EMPTY_STATEMENT

    #define STDGPU_DETAIL_HOST_EXPECTS(...) STDGPU_DETAIL_HOST_CHECK("Precondition", __VA_ARGS__)
    #define STDGPU_DETAIL_HOST_ENSURES(...) STDGPU_DETAIL_HOST_CHECK("Postcondition", __VA_ARGS__)
    #define STDGPU_DETAIL_HOST_ASSERT(...) STDGPU_DETAIL_HOST_CHECK("Assertion", __VA_ARGS__)

    #define STDGPU_DETAIL_DEVICE_CHECK(...) assert((__VA_ARGS__)) // NOLINT(hicpp-no-array-decay)

    #define STDGPU_DETAIL_DEVICE_EXPECTS(...) STDGPU_DETAIL_DEVICE_CHECK(__VA_ARGS__)
    #define STDGPU_DETAIL_DEVICE_ENSURES(...) STDGPU_DETAIL_DEVICE_CHECK(__VA_ARGS__)
    #define STDGPU_DETAIL_DEVICE_ASSERT(...) STDGPU_DETAIL_DEVICE_CHECK(__VA_ARGS__)

    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        #define STDGPU_EXPECTS(...) STDGPU_DETAIL_DEVICE_EXPECTS(__VA_ARGS__)
        #define STDGPU_ENSURES(...) STDGPU_DETAIL_DEVICE_ENSURES(__VA_ARGS__)
        #define STDGPU_ASSERT(...) STDGPU_DETAIL_DEVICE_ASSERT(__VA_ARGS__)
    #else
        #define STDGPU_EXPECTS(...) STDGPU_DETAIL_HOST_EXPECTS(__VA_ARGS__)
        #define STDGPU_ENSURES(...) STDGPU_DETAIL_HOST_ENSURES(__VA_ARGS__)
        #define STDGPU_ASSERT(...) STDGPU_DETAIL_HOST_ASSERT(__VA_ARGS__)
    #endif
#else
    #define STDGPU_EXPECTS(...) STDGPU_DETAIL_EMPTY_STATEMENT
    #define STDGPU_ENSURES(...) STDGPU_DETAIL_EMPTY_STATEMENT
    #define STDGPU_ASSERT(...) STDGPU_DETAIL_EMPTY_STATEMENT
#endif

} // namespace stdgpu

/**
 * @}
 */

#endif // STDGPU_CONTRACT_H
