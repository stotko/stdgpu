/*
 *  Copyright 2022 Patrick Stotko
 *  Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef STDGPU_NUMERIC_DETAIL_H
#define STDGPU_NUMERIC_DETAIL_H

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <type_traits>
#include <utility>

#include <stdgpu/algorithm.h>
#include <stdgpu/execution.h>
#include <stdgpu/platform.h>

#if STDGPU_BACKEND == STDGPU_BACKEND_HIP
    #include <cstddef>

    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
    #include <thrust/transform.h>
#endif

namespace stdgpu
{

namespace detail
{

#if STDGPU_BACKEND == STDGPU_BACKEND_HIP
// Reductions at or below this size are folded on the host to sidestep a rocThrust
// single-block transform_reduce hang observed on some RDNA GPUs (see
// transform_reduce_index). rocPRIM runs the single-block reduce exactly when the
// size does not exceed its reduce-config items_per_block (block_size *
// items_per_thread); above that bound the reduce is multi-block and unaffected.
// This bound was measured with rocPRIM 4.4.0 (debug_synchronous) on gfx1201: 256 *
// 16 == 4096 for both reduced element types used here (bool and index_t). Folding
// sizes up to 4096 therefore covers every size that would take the single-block
// path, and no larger reduction is ever folded onto the host.
inline constexpr index_t transform_reduce_index_host_threshold = 4096;
#endif

template <typename Iterator, typename T>
class iota_functor
{
public:
    iota_functor(Iterator begin, T value)
      : _begin(begin)
      , _value(value)
    {
    }

    STDGPU_HOST_DEVICE void
    operator()(const index_t i)
    {
        _begin[i] = _value + static_cast<T>(i);
    }

private:
    Iterator _begin;
    T _value;
};
} // namespace detail

namespace adl_barrier
{
template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
void
iota(ExecutionPolicy&& policy, Iterator begin, Iterator end, T value)
{
    for_each_index(std::forward<ExecutionPolicy>(policy),
                   static_cast<index_t>(end - begin),
                   detail::iota_functor<Iterator, T>(begin, value));
}
} // namespace adl_barrier

template <typename IndexType,
          typename ExecutionPolicy,
          typename T,
          typename BinaryFunction,
          typename UnaryFunction,
          STDGPU_DETAIL_OVERLOAD_DEFINITION_IF(is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)>
T
transform_reduce_index(ExecutionPolicy&& policy, IndexType size, T init, BinaryFunction reduce, UnaryFunction f)
{
#if STDGPU_BACKEND == STDGPU_BACKEND_HIP
    if constexpr (std::is_same_v<remove_cvref_t<ExecutionPolicy>, execution::device_policy>)
    {
        if (size <= 0)
        {
            return init;
        }

        // Small reductions hit rocThrust's single-block transform_reduce, which can
        // hang in its result-copy path on some RDNA GPUs for degenerate inputs.
        // Materialize the transformed values with a plain thrust::transform (an
        // element-wise launch that is unaffected) and fold them on the host.
        if (size <= detail::transform_reduce_index_host_threshold)
        {
            thrust::device_vector<T> values(static_cast<std::size_t>(size));
            thrust::transform(policy,
                              thrust::counting_iterator<IndexType>(0),
                              thrust::counting_iterator<IndexType>(size),
                              values.begin(),
                              f);

            thrust::host_vector<T> host_values(values);
            T result = init;
            for (std::size_t i = 0; i < host_values.size(); ++i)
            {
                result = reduce(result, host_values[i]);
            }
            return result;
        }
    }
#endif

    return thrust::transform_reduce(std::forward<ExecutionPolicy>(policy),
                                    thrust::counting_iterator<IndexType>(0),
                                    thrust::counting_iterator<IndexType>(size),
                                    f,
                                    init,
                                    reduce);
}

} // namespace stdgpu

#endif // STDGPU_NUMERIC_DETAIL_H
