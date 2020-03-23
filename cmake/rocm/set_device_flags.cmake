
# NOTE ROCm can auto-detect the GPU, so the list below is overly pessimistic.
# NOTE However, this list of hard-coded targets allows cross-compilation and compilation on systems without AMD GPUs.
list(APPEND STDGPU_ROCM_AMDGPU_TARGETS "gfx803" "gfx900" "gfx906" "gfx908")

set(STDGPU_ROCM_HAVE_SUITABLE_GPU FALSE)

foreach(STDGPU_ROCM_TARGET IN LISTS STDGPU_ROCM_AMDGPU_TARGETS)
    list(APPEND STDGPU_DEVICE_LINK_FLAGS "--amdgpu-target=${STDGPU_ROCM_TARGET}")
    message(STATUS "  Enabled compilation for AMDGPU target ${STDGPU_ROCM_TARGET}")
    set(STDGPU_ROCM_HAVE_SUITABLE_GPU TRUE)
endforeach()

if(NOT STDGPU_ROCM_HAVE_SUITABLE_GPU)
    message(FATAL_ERROR "  No ROCm-capable GPU detected")
endif()


# FIXME These are needed to suppress some weird warnings/errors
string(APPEND STDGPU_DEVICE_FLAGS " -Wno-invalid-noreturn")

# Apply compiler flags
string(APPEND CMAKE_CXX_FLAGS ${STDGPU_DEVICE_FLAGS})

message(STATUS "Created device flags : ${STDGPU_DEVICE_FLAGS}")
message(STATUS "Created device link flags : ${STDGPU_DEVICE_LINK_FLAGS}")
message(STATUS "Building with CXX flags : ${CMAKE_CXX_FLAGS}")
