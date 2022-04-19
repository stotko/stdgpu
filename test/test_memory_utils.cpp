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

#include <test_memory_utils.h>

namespace test_utils
{

void
allocator_statistics::reset()
{
    default_constructions = 0;
    copy_constructions = 0;
    destructions = 0;
}

allocator_statistics&
get_allocator_statistics()
{
    static allocator_statistics stats;
    return stats;
}

} // namespace test_utils
