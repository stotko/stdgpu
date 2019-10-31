
if(NOT MSVC)
    string(APPEND STDGPU_HOST_FLAGS " -Wall")
    string(APPEND STDGPU_HOST_FLAGS " -pedantic")
    string(APPEND STDGPU_HOST_FLAGS " -Wextra")
    string(APPEND STDGPU_HOST_FLAGS " -O3")
else()
    #string(APPEND STDGPU_HOST_FLAGS " /W3") # or /W4 depending on how useful this is
    #string(APPEND STDGPU_HOST_FLAGS " /O2")
endif()

# Apply compiler flags
string(APPEND CMAKE_CXX_FLAGS ${STDGPU_HOST_FLAGS})

message(STATUS "Created host flags : ${STDGPU_HOST_FLAGS}")
message(STATUS "Building with CXX flags : ${CMAKE_CXX_FLAGS}")
