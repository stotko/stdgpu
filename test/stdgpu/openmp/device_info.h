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

#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#include <chrono>
#include <cstddef>
#include <random>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <stdgpu/cstddef.h>



namespace stdgpu
{

namespace openmp
{

/**
 * \brief Prints the technical data of the currently used device
 */
void
print_device_information();

} // namespace openmp

} // namespace stdgpu



#endif // DEVICE_INFO_H
