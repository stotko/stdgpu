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

#include <gtest/gtest.h>

#include <limits>

#include <stdgpu/limits.h>
#include <stdgpu/platform.h>



class stdgpu_limits : public ::testing::Test
{
    protected:
        // Called before each test
        void SetUp() override
        {

        }

        // Called after each test
        void TearDown() override
        {

        }

};


// Explicit template instantiations
namespace stdgpu
{

// Instantiation of specialized templates emit no-effect warnings with Clang
/*
template
struct numeric_limits<bool>;

template
struct numeric_limits<char>;

template
struct numeric_limits<signed char>;

template
struct numeric_limits<unsigned char>;

template
struct numeric_limits<wchar_t>;

template
struct numeric_limits<char16_t>;

template
struct numeric_limits<char32_t>;

template
struct numeric_limits<short>;

template
struct numeric_limits<unsigned short>;

template
struct numeric_limits<int>;

template
struct numeric_limits<unsigned int>;

template
struct numeric_limits<long>;

template
struct numeric_limits<unsigned long>;

template
struct numeric_limits<long long>;

template
struct numeric_limits<unsigned long long>;

template
struct numeric_limits<float>;

template
struct numeric_limits<double>;

template
struct numeric_limits<long double>;
*/

} // namespace stdgpu


template <typename Type>
void check()
{
    EXPECT_EQ(stdgpu::numeric_limits<Type>::min(),          std::numeric_limits<Type>::min());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::max(),          std::numeric_limits<Type>::max());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::lowest(),       std::numeric_limits<Type>::lowest());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::epsilon(),      std::numeric_limits<Type>::epsilon());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::round_error(),  std::numeric_limits<Type>::round_error());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::infinity(),     std::numeric_limits<Type>::infinity());
    EXPECT_EQ(stdgpu::numeric_limits<Type>::is_specialized, std::numeric_limits<Type>::is_specialized);
    EXPECT_EQ(stdgpu::numeric_limits<Type>::is_signed,      std::numeric_limits<Type>::is_signed);
    EXPECT_EQ(stdgpu::numeric_limits<Type>::is_integer,     std::numeric_limits<Type>::is_integer);
    EXPECT_EQ(stdgpu::numeric_limits<Type>::is_exact,       std::numeric_limits<Type>::is_exact);
    EXPECT_EQ(stdgpu::numeric_limits<Type>::digits,         std::numeric_limits<Type>::digits);
    EXPECT_EQ(stdgpu::numeric_limits<Type>::radix,          std::numeric_limits<Type>::radix);
}


TEST_F(stdgpu_limits, bool)
{
    check<bool>();
}


TEST_F(stdgpu_limits, char)
{
    check<char>();
}


TEST_F(stdgpu_limits, signed_char)
{
    check<signed char>();
}


TEST_F(stdgpu_limits, unsigned_char)
{
    check<unsigned char>();
}


TEST_F(stdgpu_limits, wchar_t)
{
    check<wchar_t>();
}


TEST_F(stdgpu_limits, char16_t)
{
    check<char16_t>();
}


TEST_F(stdgpu_limits, char32_t)
{
    check<char32_t>();
}


TEST_F(stdgpu_limits, short)
{
    check<short>();
}


TEST_F(stdgpu_limits, unsigned_short)
{
    check<unsigned short>();
}


TEST_F(stdgpu_limits, int)
{
    check<int>();
}


TEST_F(stdgpu_limits, unsigned_int)
{
    check<unsigned int>();
}


TEST_F(stdgpu_limits, long)
{
    check<long>();
}


TEST_F(stdgpu_limits, unsigned_long)
{
    check<unsigned long>();
}


TEST_F(stdgpu_limits, long_long)
{
    check<long long>();
}


TEST_F(stdgpu_limits, unsigned_long_long)
{
    check<unsigned long long>();
}


TEST_F(stdgpu_limits, float)
{
    check<float>();
}


TEST_F(stdgpu_limits, double)
{
    check<double>();
}


TEST_F(stdgpu_limits, long_double)
{
    check<long double>();
}


class NonArithmeticType
{
    public:
        inline STDGPU_HOST_DEVICE bool
        operator==(const NonArithmeticType& other) const
        {
            return x == other.x;
        }

    private:
        int x = 0;
};

TEST_F(stdgpu_limits, NonArithmeticType)
{
    check<NonArithmeticType>();
}

