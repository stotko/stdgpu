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

#include <type_traits>

#include <stdgpu/config.h>
#include <stdgpu/contract.h>

class stdgpu_contract : public ::testing::Test
{
protected:
    // Called before each test
    void
    SetUp() override
    {
    }

    // Called after each test
    void
    TearDown() override
    {
    }
};

TEST_F(stdgpu_contract, expects_host_value)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile bool true_value = true;

    STDGPU_EXPECTS(true_value);

    volatile bool false_value = false;

    EXPECT_DEATH(STDGPU_EXPECTS(false_value), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, expects_host_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile int value_1 = 42; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    volatile int value_2 = 24; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    STDGPU_EXPECTS(value_1 == 42 && value_2 > 0);

    EXPECT_DEATH(STDGPU_EXPECTS(value_1 != 42 || value_2 <= 0), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, expects_host_comma_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    STDGPU_EXPECTS(std::is_same_v<int, int>); // NOLINT(hicpp-static-assert,misc-static-assert,cert-dcl03-c)

    // NOLINTNEXTLINE(hicpp-no-array-decay,hicpp-avoid-goto,hicpp-static-assert,misc-static-assert,cert-dcl03-c)
    EXPECT_DEATH(STDGPU_EXPECTS(std::is_same_v<int, float>), "");
#endif
}

TEST_F(stdgpu_contract, ensures_host_value)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile bool true_value = true;

    STDGPU_ENSURES(true_value);

    volatile bool false_value = false;

    EXPECT_DEATH(STDGPU_ENSURES(false_value), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, ensures_host_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile int value_1 = 42; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    volatile int value_2 = 24; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    STDGPU_ENSURES(value_1 == 42 && value_2 > 0);

    EXPECT_DEATH(STDGPU_ENSURES(value_1 != 42 || value_2 <= 0), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, ensures_host_comma_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    STDGPU_ENSURES(std::is_same_v<int, int>); // NOLINT(hicpp-static-assert,misc-static-assert,cert-dcl03-c)

    // NOLINTNEXTLINE(hicpp-no-array-decay,hicpp-avoid-goto,hicpp-static-assert,misc-static-assert,cert-dcl03-c)
    EXPECT_DEATH(STDGPU_ENSURES(std::is_same_v<int, float>), "");
#endif
}

TEST_F(stdgpu_contract, assert_host_value)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile bool true_value = true;

    STDGPU_ASSERT(true_value);

    volatile bool false_value = false;

    EXPECT_DEATH(STDGPU_ASSERT(false_value), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, assert_host_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    volatile int value_1 = 42; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    volatile int value_2 = 24; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    STDGPU_ASSERT(value_1 == 42 && value_2 > 0);

    EXPECT_DEATH(STDGPU_ASSERT(value_1 != 42 || value_2 <= 0), ""); // NOLINT(hicpp-no-array-decay,hicpp-avoid-goto)
#endif
}

TEST_F(stdgpu_contract, assert_host_comma_expression)
{
#if STDGPU_ENABLE_CONTRACT_CHECKS
    STDGPU_ASSERT(std::is_same_v<int, int>); // NOLINT(hicpp-static-assert,misc-static-assert,cert-dcl03-c)

    // NOLINTNEXTLINE(hicpp-no-array-decay,hicpp-avoid-goto,hicpp-static-assert,misc-static-assert,cert-dcl03-c)
    EXPECT_DEATH(STDGPU_ASSERT(std::is_same_v<int, float>), "");
#endif
}
