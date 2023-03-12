Iterating over Arrays and Containers {#iterator}
====================================


# Motivation {#iterator_overview}

The iterator concept is one of the core aspects of the Standard Template Library (STL). Most C++ programmers are familiar with this concept and can easily write algorithms with it. The thrust library aims to provide the STL functionality also for device arrays and vectors. This includes the convenient iterator syntax. Consider the following STL example:

```cpp
    #include <algortihm>
    #include <functional>
    #include <vector>

    std::vector<float> vector(1000);

    // Fill it with something useful

    std::sort(vector.begin(), vector.end());            // C++98
    std::sort(std::begin(vector), std::end(vector));    // C++11
```

In modern C++, the latter more recent version of begin and end should be used. Semantically, they are identical. thrust provides a similar syntax for its containers:

```cpp
    #include <thrust/device_vector.h>
    #include <thrust/sort.h>

    thrust::device_vector<float> device_vector(1000);

    // Fill it with something useful

    thrust::sort(thrust::device, vector.begin(), vector.end());
```

The differences to the STL are mostly related to the more generic setting. Although thrust is able to automatically infer whether the vector is allocated on the host or device, it is advisable to clearly state that sorting should be done on the device.

It is also possible to pass raw pointers to thrust algorithms, but the syntax gets intrusive:

```cpp
    #include <thrust/device_ptr.h>
    #include <thrust/sort.h>

    #include <stdgpu/memory.h>

    float* device_array = createDeviceArray<float>(1000);

    // Fill it with something useful

    thrust::sort(thrust::device, thrust::device_pointer_cast(device_array), thrust::device_pointer_cast(device_array + 1000));

    destroyDeviceArray<float>(device_array);
```

The intent of casting to the thrust API is clear, but very verbose. Furthermore, the size of the array must be known and explicitly stated to compute the iterator pointing to the end of the array.


# Iterator API {#iterator_api}

Similar to what the [memory management API](#memory) provides, there is also an API to avoid boilerplate code such as in the example above. It can be considered as a natural extension to how thrust and STL perform in C++11:

```cpp
    #include <thrust/sort.h>

    #include <stdgpu/memory.h>
    #include <stdgpu/iterator.h>

    float* device_array = createDeviceArray<float>(1000);

    // Fill it with something useful

    thrust::sort(stdgpu::device_begin(device_array), stdgpu::device_end(device_array));

    destroyDeviceArray<float>(device_array);
```

Compare this systax to the C++11 version of the STL call. This becomes possible by the internal leak check which now provides the required size information. Therefore, stdgpu::device_end can query the size of the given array and return a pointer to the end. Furthermore, the functions check whether the array is allocated on the host or device to avoid mismatches. Iterators are defined for both host and device arrays. In addition, the const versions of them are also defined. Consider the following C++14 STL example:

```cpp
    #include <algortihm>
    #include <vector>

    std::vector<float> vector(1000);
    std::vector<float> vector_out(1000);

    // Fill it with something useful

    std::transform(std::cbegin(vector), std::cend(vector), std::begin(vector_out), std::negate<float>());  // C++14
```

The device version with the memory management API is almost identical:

```cpp
    #include <thrust/transform.h>
    #include <thrust/functional.h>

    #include <stdgpu/memory.h>
    #include <stdgpu/iterator.h>

    float* device_array     = createDeviceArray<float>(1000);
    float* device_array_out = createDeviceArray<float>(1000);

    // Fill it with something useful

    thrust::transform(stdgpu::device_cbegin(device_array), stdgpu::device_cend(device_array), stdgpu::device_begin(device_array_out), thrust::negate<float>());

    destroyDeviceArray<float>(device_array);
    destroyDeviceArray<float>(device_array_out);
```

The combination of both the memory management and the iterator API provides a very powerful interface to interact with thrust as well as kernels in a fast, safe and intuitive way.
