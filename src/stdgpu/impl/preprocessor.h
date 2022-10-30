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

#ifndef STDGPU_PREPROCESSOR_H
#define STDGPU_PREPROCESSOR_H

namespace stdgpu::detail
{

#define STDGPU_DETAIL_CAT2_DIRECT(A, B) A##B
#define STDGPU_DETAIL_CAT2(A, B) STDGPU_DETAIL_CAT2_DIRECT(A, B)

#define STDGPU_DETAIL_CAT3(A, B, C) STDGPU_DETAIL_CAT2(A, STDGPU_DETAIL_CAT2(B, C))

} // namespace stdgpu::detail

#endif // STDGPU_PREPROCESSOR_H
