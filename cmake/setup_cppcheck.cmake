function(stdgpu_setup_cppcheck STDGPU_OUTPUT_PROPERTY_CPPCHECK)
    find_program(STDGPU_CPPCHECK
                 NAMES
                 "cppcheck")

    if(NOT STDGPU_CPPCHECK)
        message(FATAL_ERROR "cppcheck not found.")
    endif()

    set(${STDGPU_OUTPUT_PROPERTY_CPPCHECK} "${STDGPU_CPPCHECK}" "--enable=warning,style,performance,portability" "--force" "--inline-suppr" "--quiet")

    if(NOT DEFINED STDGPU_TREAT_WARNINGS_AS_ERRORS)
        message(FATAL_ERROR "STDGPU_TREAT_WARNINGS_AS_ERRORS not defined.")
    endif()

    if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
        list(APPEND ${STDGPU_OUTPUT_PROPERTY_CPPCHECK} "--error-exitcode=1")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_PROPERTY_CPPCHECK} ${${STDGPU_OUTPUT_PROPERTY_CPPCHECK}} PARENT_SCOPE)
endfunction()
