/*
 *  Copyright 2021 Patrick Stotko
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

#ifndef STDGPU_PLATFORM_CHECK_H
#define STDGPU_PLATFORM_CHECK_H


#include <stdgpu/config.h>

//! @cond Doxygen_Suppress
#define STDGPU_BACKEND_PLATFORM_CHECK_HEADER <stdgpu/STDGPU_BACKEND_DIRECTORY/platform_check.h> // NOLINT(bugprone-macro-parentheses,misc-macro-parentheses)
// cppcheck-suppress preprocessorErrorDirective
#include STDGPU_BACKEND_PLATFORM_CHECK_HEADER
#undef STDGPU_BACKEND_PLATFORM_CHECK_HEADER
//! @endcond



#endif // STDGPU_PLATFORM_CHECK_H
