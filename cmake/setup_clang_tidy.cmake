function(stdgpu_setup_clang_tidy STDGPU_OUTPUT_PROPERTY_CLANG_TIDY)
    find_package(ClangTidy REQUIRED)

    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "${CLANG_TIDY_EXECUTABLE}")

    if(NOT DEFINED STDGPU_TREAT_WARNINGS_AS_ERRORS)
        message(FATAL_ERROR "STDGPU_TREAT_WARNINGS_AS_ERRORS not defined.")
    endif()

    if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
        list(APPEND ${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "-warnings-as-errors=*")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} ${${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY}} PARENT_SCOPE)
endfunction()
