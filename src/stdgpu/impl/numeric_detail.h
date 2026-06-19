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
    #include <hip/hip_runtime.h>

    #include <stdgpu/cstddef.h>
    #include <stdgpu/hip/impl/error.h>
#endif

namespace stdgpu
{

namespace detail
{

#if STDGPU_BACKEND == STDGPU_BACKEND_HIP
template <index_t block_size, typename IndexType, typename T, typename BinaryFunction, typename UnaryFunction>
__global__ void
transform_reduce_index_kernel(IndexType size, T init, BinaryFunction reduce, UnaryFunction f, T* block_results)
{
    __shared__ T shared[block_size];

    index_t tid = static_cast<index_t>(threadIdx.x);
    IndexType stride = static_cast<IndexType>(block_size) * static_cast<IndexType>(gridDim.x);

    T value = init;
    for (IndexType i = static_cast<IndexType>(blockIdx.x) * static_cast<IndexType>(block_size) + tid; i < size;
         i += stride)
    {
        value = reduce(value, f(i));
    }
    shared[tid] = value;
    __syncthreads();

    for (index_t offset = block_size / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            shared[tid] = reduce(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_results[blockIdx.x] = shared[0];
    }
}

// Hand-written device reduction used in place of thrust::transform_reduce for the device execution policy.
// rocThrust's single-block transform_reduce wedges its internal hipMemcpyWithStream result copy on some
// RDNA GPUs for degenerate inputs; a plain kernel plus a plain hipMemcpy D2H avoids that path while
// computing the identical reduction (same init, same associative BinaryFunction).
template <typename IndexType, typename T, typename BinaryFunction, typename UnaryFunction>
T
transform_reduce_index_device(IndexType size, T init, BinaryFunction reduce, UnaryFunction f)
{
    if (size <= 0)
    {
        return init;
    }

    constexpr index_t block_size = 256;
    constexpr index_t max_blocks = 64;
    index_t blocks = static_cast<index_t>((static_cast<std::int64_t>(size) + block_size - 1) / block_size);
    blocks = (blocks < 1) ? 1 : ((blocks > max_blocks) ? max_blocks : blocks);

    T* block_results = nullptr;
    STDGPU_HIP_SAFE_CALL(hipMalloc(&block_results, static_cast<std::size_t>(blocks) * sizeof(T)));

    transform_reduce_index_kernel<block_size>
            <<<static_cast<unsigned int>(blocks), block_size>>>(size, init, reduce, f, block_results);
    STDGPU_HIP_SAFE_CALL(hipGetLastError());
    STDGPU_HIP_SAFE_CALL(hipDeviceSynchronize());

    T host_results[max_blocks];
    STDGPU_HIP_SAFE_CALL(
            hipMemcpy(host_results, block_results, static_cast<std::size_t>(blocks) * sizeof(T), hipMemcpyDeviceToHost));
    STDGPU_HIP_SAFE_CALL(hipFree(block_results));

    T result = init;
    for (index_t i = 0; i < blocks; ++i)
    {
        result = reduce(result, host_results[i]);
    }
    return result;
}
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
        return detail::transform_reduce_index_device(size, init, reduce, f);
    }
    else
#endif
    {
        return thrust::transform_reduce(std::forward<ExecutionPolicy>(policy),
                                        thrust::counting_iterator<IndexType>(0),
                                        thrust::counting_iterator<IndexType>(size),
                                        f,
                                        init,
                                        reduce);
    }
}

} // namespace stdgpu

#endif // STDGPU_NUMERIC_DETAIL_H
