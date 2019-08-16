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

#ifndef STDGPU_UTILITY_DETAIL_H
#define STDGPU_UTILITY_DETAIL_H



namespace stdgpu
{

template< class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>& t) noexcept
{
    return static_cast<T&&>(t);
}


template <class T>
constexpr STDGPU_HOST_DEVICE T&&
forward(std::remove_reference_t<T>&& t) noexcept
{
    return static_cast<T&&>(t);
}


template<class T>
constexpr STDGPU_HOST_DEVICE std::remove_reference_t<T>&&
move(T&& t) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(t);
}

} // namespace stdgpu



#endif // STDGPU_UTILITY_DETAIL_H
