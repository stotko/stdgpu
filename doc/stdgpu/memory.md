Memory Management {#memory}
=================


# Motivation {#memory_overview}

Memory management in C and C++ is one of the main aspects a developer should care about. Usually, containers such as std::vector are sufficient for most use cases. However, all these convenient containers are unfortunately not directly supported on the GPU side. There has been some effort to provide a GPU version of the Standard Template Library (STL), but the solutions are still quite limited. For instance, the thrust library (which is delivered with CUDA by default) has the following limitations:

- There are only CPU (host) and GPU (device) versions of std::vector called thrust::host_vector and thrust::device_vector, but other containers are not present.
- It is not possible to directly pass a thrust::device_vector to a kernel.
- The automatic garbage collection of thrust::device_vector avoids memory leaks but may lead to unwanted copies and waste of memory bandwidth.

Typically, applications involving the GPU are performance critical and should be well optimized. On the other hand, thrust allows to develop code at a fast pace when sticking to their API. Therefore, most developers use raw pointers to achieve maximum performance. The drawback is that this requires a C-like API to allocate and free host and device memory which is quite intrusive and prone to errors.


# Memory API {#memory_api}

In order to solve this problem, a simple and consistent wrapper API around the memory management functions is defined. The goal is to reduce boilerplate code and give the user strong guarantees about the requested operations.


## Creating and Destroying Arrays {#memory_create_destroy}

The simplest operation when dealing with dynamically allocated arrays is to create such an array. This can be done in the following way:

```cpp
    #include <stdgpu/memory.h>

    float* device_float_vector = createDeviceArray<float>(1000, 42.0f);
    float* host_float_vector = createHostArray<float>(1000, 42.0f);
```

Here, two arrays of length 1000 are created, one on the host and one on the device, and filled with the value 42.0f. Compared to traditional C and C++ allocations, these functions guarantee that the allocated memory is initialized with a well-defined state. The value parameter is optional. In case, no value is given, a default constructed object is used, i.e. float() which equals to 0.0f.

When memory is allocated, it must be freed later at some time:

```cpp
    #include <stdgpu/memory.h>

    // Define device_float_vector and host_float_vector

    destroyDeviceArray<float>(device_float_vector);
    destroyHostArray<float>(host_float_vector);
```

Although these functions are a bit more consistent than the usual memory management functions, the additional overhead of defining this wrapper might not be worth the effort. However, there are several more implicit guarantees:

- If the allocation of an array fails, a warning is printed and a null pointer is returned to the user.
- The destroy{Host,Device}Array functions check whether the array is valid, i.e. not a null pointer, and only then free it. Afterwards, it overwrites the freed pointer to a null pointer. This avoids double free errors leads again to a well-defined state.
- Internally, a leak checker maintains a list of allocated arrays. If the user forgets to free an array, a warning can be prompted that some memory is leaking, so that the user can fix this problem easily.


## Copying Arrays between Host and Device {#memory_copy}

Creating and destroying arrays is only one step to efficiently handle memory. Data usually become available on the host, but should be processed on the device for performance reasons, and in the end stored on the host again. Consequently, copying arrays between the host and the device is necessary. This can be done by:

```cpp
    #include <stdgpu/memory.h>

    float* host_float_vector;   // Create this and set some values
    float* device_float_vector; // Create this

    copyHost2DeviceArray<float>(host_float_vector, 1000, device_float_vector);

    // Do something useful with device_float_vector

    copyDevice2HostArray<float>(device_float_vector, 1000, host_float_vector);
```

Here, the first 1000 values of host_float_vector are copyied to device_float_vector and later on vice versa. If the array to which should be copied is not allocated so far, then one can use these functions to unify allocation and copy:

```cpp
    #include <stdgpu/memory.h>

    float* host_float_vector;   // Create this and set some values

    float* device_float_vector = copyCreateHost2DeviceArray<float>(host_float_vector, 1000);
```

All of these copy functions share strong guarantees. Having the internal leak checker, the copy functions check if the arrays are indeed allocated on the host or device. This avoids accidental mismatches. Furthermore, the size of the arrays are checked to prevent copying elements out of the allocated bounds of both arrays.

It is very important to note that these guarantees can only be fulfilled if the arrays have been allocated by this API. External arrays or pointers to stack objects can also be used with this API. However, the checks need to be disabled in this situation:

```cpp
    #include <stdgpu/memory.h>

    float host_value = 42.0f;

    float* device_value_pointer = copyCreateHost2DeviceArray<float>(&host_value, 1, MemoryCopy::NO_CHECK);
```

Please keep in mind, that if the functions are called in this way, it is then your responsibility to make sure that this operation succeeds.
