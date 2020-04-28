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
 * \file stdgpu/contract.h
 */

#include <cstdio>
#include <cassert>
#include <exception>

#include <stdgpu/config.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>



namespace stdgpu
{

/**
 * \def STDGPU_EXPECTS(condition)
 * \hideinitializer
 * \brief A macro to define pre-conditions for both host and device
 */
/**
 * \def STDGPU_ENSURES(condition)
 * \hideinitializer
 * \brief A macro to define post-conditions for both host and device
 */
/**
 * \def STDGPU_ASSERT(condition)
 * \hideinitializer
 * \brief A macro to define in-body conditions for both host and device
 */
#if STDGPU_ENABLE_CONTRACT_CHECKS
    #define STDGPU_DETAIL_HOST_CHECK(type, condition) \
        if (!(condition)) \
        { \
            printf("stdgpu : " type " failure :\n" \
                   "  File      : %s:%d\n" \
                   "  Function  : %s\n" \
                   "  Condition : %s\n", \
                   __FILE__, __LINE__, static_cast<const char*>(STDGPU_FUNC), #condition); \
            std::terminate(); \
        }

    #define STDGPU_DETAIL_HOST_EXPECTS(condition) STDGPU_DETAIL_HOST_CHECK("Precondition", condition)
    #define STDGPU_DETAIL_HOST_ENSURES(condition) STDGPU_DETAIL_HOST_CHECK("Postcondition", condition)
    #define STDGPU_DETAIL_HOST_ASSERT(condition) STDGPU_DETAIL_HOST_CHECK("Assertion", condition)

    // FIXME:
    // HIP's device assert() function does not seem to override/overload the host compiler version.
    // Even using HIP's device assert() function implementation directly results in linker errors.
    // Thus, disable contract checks until a better workaround/fix is found.
    #if STDGPU_BACKEND == STDGPU_BACKEND_HIP
        #define STDGPU_DETAIL_WORKAROUND_ASSERT(condition)
    #else
        #define STDGPU_DETAIL_WORKAROUND_ASSERT(condition) assert(condition)
    #endif

    #define STDGPU_DETAIL_DEVICE_EXPECTS(condition) STDGPU_DETAIL_WORKAROUND_ASSERT(condition)
    #define STDGPU_DETAIL_DEVICE_ENSURES(condition) STDGPU_DETAIL_WORKAROUND_ASSERT(condition)
    #define STDGPU_DETAIL_DEVICE_ASSERT(condition) STDGPU_DETAIL_WORKAROUND_ASSERT(condition)

    #if STDGPU_CODE == STDGPU_CODE_DEVICE
        #define STDGPU_EXPECTS(condition) STDGPU_DETAIL_DEVICE_EXPECTS(condition)
        #define STDGPU_ENSURES(condition) STDGPU_DETAIL_DEVICE_ENSURES(condition)
        #define STDGPU_ASSERT(condition) STDGPU_DETAIL_DEVICE_ASSERT(condition)
    #else
        #define STDGPU_EXPECTS(condition) STDGPU_DETAIL_HOST_EXPECTS(condition)
        #define STDGPU_ENSURES(condition) STDGPU_DETAIL_HOST_ENSURES(condition)
        #define STDGPU_ASSERT(condition) STDGPU_DETAIL_HOST_ASSERT(condition)
    #endif
#else
    #define STDGPU_EXPECTS(condition)
    #define STDGPU_ENSURES(condition)
    #define STDGPU_ASSERT(condition)
#endif

} // namespace stdgpu



#endif // STDGPU_CONTRACT_H
