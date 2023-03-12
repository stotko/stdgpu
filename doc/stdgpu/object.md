Container Objects {#object}
=================

# Motivation {#object_overview}

In order to bridge the gap between GPU and CPU programming, the memory management API allows to handle arrays in an efficient and reliable way (see \ref memory). However, this is not sufficient even for most projects which require at least one layer hiding all the computations and memory management operations. Therefore, semantically coherent data (e.g. arrays) should be packed together into a class and processed by the public interface of the class. One category of such classes are containers including std::vector. In the context of GPU programming, array-based data structures are prefered and easier to implement. The thrust library for example only provides a thrust::host_vector and thrust::device_vector class because other structures such as std::unordered_map, std::list, etc. are very difficult to port to the GPU without sacrificing some important properties. While the containers defined in this library also have some limitations, they are still easy to use and robust.


# Defining Host and Device Container Objects {#object_api}

As mentioned above, a further abstraction layer to simplify data management is needed. This requires another API to avoid boilerplate code and redudancy. So far, host and device arrays has been defined as the generalization of arrays to CPU and GPU memory. Consequently, host device objects now generalize the traditional class objects. Consider the following class:

```cpp
    class MyClass
    {
        public:
            MyClass()
            {
                this->array = nullptr;
                this->size = 0;
            }

            MyClass(const int size)
            {
                this->array = new float[size];
                this->size = size;
            }

            ~MyClass()
            {
                delete[] array;
                size = 0;
            }

            void function(const int parameter) const
            {
                // Do something useful with array
            }

        private:
            float* array;
            int size;
    };
```

It wraps an array of type float including the size and provides some interaction interface through the member function. There are two constructors for this class. The first is simply the default constructor which should set the object to an empty state. The other constructor allocates the array with the given size. Finally, the destructor cleans them up. This design is quite problematic since copy and move constructors are not considered here which can result to double free errors and memory leaks. Furthermore, this design does not scale to the GPU and a new API must be used. Consider this API on the aforementioned example:

```cpp
    class MyHostDeviceObjectClass
    {
        public:
            MyHostDeviceObjectClass()
            {
                this->_array = nullptr;
                this->_size = 0;
            }

            static MyHostDeviceObjectClass createDeviceObject(const int size)
            {
                MyHostDeviceObjectClass result;

                result._array = createDeviceArray<float>(size);
                result._size = size;

                return result;
            }

            static void destroyDeviceObject(MyHostDeviceObjectClass& device_object)
            {
                destroyDeviceArray<float>(device_object._array);
                device_object._size = 0;
            }

            static MyHostDeviceObjectClass createHostObject(const int size)
            {
                MyHostDeviceObjectClass result;

                result._array = createHostArray<float>(size);
                result._size = size;

                return result;
            }

            static void destroyHostObject(MyHostDeviceObjectClass& host_object)
            {
                destroyHostArray<float>(host_object._array);
                host_object._size = 0;
            }

            void function(const int parameter) const
            {
                // Do something useful with array
            }

        private:
            float* _array;
            int _size;
    };
```

Note that this interface is very similar to the [host device array interface](#memory). An object can now be easily created and destroyed as follows:

```cpp
    MyClass device_object = MyClass::createDeviceObject(1000);
    MyClass hoste_object = MyClass::createHostObject(1000);

    // Do something with device_object and host_object

    MyClass::destroyDeviceObject(device_object);
    MyClass::destroyHostObject(host_object);
```

In order to match the capabilities of the host device arrays, copy functions can defined in the same manner and used as:

```cpp
    MyClass device_object = MyClass::createDeviceObject(1000);

    // Do something with device_object

    MyClass host_object = MyClass::copyCreateDevice2HostObject(device_object);

    // Do something with host_object

    MyClass::destroyDeviceObject(device_object);
    MyClass::destroyHostObject(host_object);
```

Compared to arrays, an object always knows its size, so it is not necessary to also pass it as a parameter. This design is used to define the containers in this library.
