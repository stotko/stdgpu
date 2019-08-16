// Check for GPUs present and their compute capability
// based on http://stackoverflow.com/questions/2285185/easiest-way-to-test-for-existence-of-cuda-capable-gpu-from-cmake/2297877#2297877 (Christopher Bruns)

#include <stdio.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <set>
#include <sstream>

int main()
{
    int deviceCount;
    std::set<std::string> computeCapabilities;

    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
    {
        deviceCount = 0;
    }

    /* machines with no GPUs can still report one emulation device */
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp currentProperties;
        cudaGetDeviceProperties(&currentProperties, device);

        /* 9999 means emulation only */
        if (currentProperties.major != 9999)
        {
            std::stringstream ss;
            ss << currentProperties.major;
            ss << currentProperties.minor;

            computeCapabilities.insert(ss.str());
        }
    }

    /* don't just return the number of gpus, because other runtime cuda
    errors can also yield non-zero return values */
    for (std::set<std::string>::const_iterator it = computeCapabilities.begin(); it != computeCapabilities.end(); ++it)
    {
        // Add a semicolon if we have already printed some output.
        if(it != computeCapabilities.begin()) std::cout << ';';
        std::cout << *it;
    }

    return computeCapabilities.size() == 0; /* 0 devices -> failure */
}
