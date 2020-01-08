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

#include <gtest/gtest.h>

#include <thrust/functional.h>

#include <stdgpu/ranges.h>



class stdgpu_ranges : public ::testing::Test
{
    protected:
        // Called before each test
        virtual void SetUp()
        {

        }

        // Called after each test
        virtual void TearDown()
        {

        }

};


// Explicit template instantiations
namespace stdgpu
{

template
class device_range<int>;

template
class host_range<int>;

template
class transform_range<device_range<int>, thrust::identity<int>>;

} // namespace stdgpu


/*
TEST_F(stdgpu_ranges, test_name)
{

}
*/


