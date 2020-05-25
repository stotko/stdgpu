/*
 *  Copyright 2020 Patrick Stotko
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

#ifndef STDGPU_OPENMP_PLATFORM_H
#define STDGPU_OPENMP_PLATFORM_H



namespace stdgpu
{
namespace openmp
{

/**
 * \def STDGPU_OPENMP_HOST_DEVICE
 * \brief Platform-independent host device function annotation
 */
#define STDGPU_OPENMP_HOST_DEVICE


/**
 * \def STDGPU_OPENMP_DEVICE_ONLY
 * \brief Platform-independent device function annotation
 */
#define STDGPU_OPENMP_DEVICE_ONLY


/**
 * \def STDGPU_OPENMP_CONSTANT
 * \brief Platform-independent constant variable annotation
 */
#define STDGPU_OPENMP_CONSTANT


/**
 * \def STDGPU_OPENMP_IS_DEVICE_CODE
 * \brief Platform-independent device code detection
 */
#define STDGPU_OPENMP_IS_DEVICE_CODE 1


/**
 * \def STDGPU_OPENMP_IS_DEVICE_COMPILED
 * \brief Platform-independent device compilation detection
 */
#define STDGPU_OPENMP_IS_DEVICE_COMPILED 1


} // namespace openmp

} // namespace stdgpu



#endif // STDGPU_OPENMP_PLATFORM_H
