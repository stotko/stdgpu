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

#include <stdgpu/device.h>

#include <stdgpu/platform.h>

#include STDGPU_DETAIL_BACKEND_HEADER(device.h)

namespace stdgpu
{

void
print_device_information()
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::print_device_information();
}

} // namespace stdgpu
