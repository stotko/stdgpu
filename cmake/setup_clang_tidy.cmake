function(stdgpu_setup_clang_tidy STDGPU_OUTPUT_PROPERTY_CLANG_TIDY)
    find_program(STDGPU_CLANG_TIDY
                 NAMES
                 "clang-tidy")

    if(NOT STDGPU_CLANG_TIDY)
        message(FATAL_ERROR "clang-tidy not found.")
    endif()

    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "${STDGPU_CLANG_TIDY}")

    if(NOT DEFINED STDGPU_TREAT_WARNINGS_AS_ERRORS)
        message(FATAL_ERROR "STDGPU_TREAT_WARNINGS_AS_ERRORS not defined.")
    endif()

    if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
        list(APPEND ${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} "-warnings-as-errors=*")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY} ${${STDGPU_OUTPUT_PROPERTY_CLANG_TIDY}} PARENT_SCOPE)
endfunction()
