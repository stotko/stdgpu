function(stdgpu_setup_clang_tidy STDGPU_OUTPUT_PROPERTY_CLANG_TIDY)
    find_package(ClangTidy REQUIRED)

    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "${CLANG_TIDY_EXECUTABLE}")

    if(NOT DEFINED STDGPU_COMPILE_WARNING_AS_ERROR)
        message(FATAL_ERROR "STDGPU_COMPILE_WARNING_AS_ERROR not defined.")
    endif()

    # Explicitly set the C++ standard
    list(APPEND ${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "-extra-arg=-std=c++17")

    if(STDGPU_COMPILE_WARNING_AS_ERROR)
        list(APPEND ${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "-warnings-as-errors=*")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} ${${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY}} PARENT_SCOPE)
endfunction()
