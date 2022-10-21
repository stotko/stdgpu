function(stdgpu_setup_cppcheck STDGPU_OUTPUT_PROPERTY_CPPCHECK)
    find_package(Cppcheck REQUIRED)

    # Do not enable noisy "style" checks
    set(${STDGPU_OUTPUT_PROPERTY_CPPCHECK} "${CPPCHECK_EXECUTABLE}" "--enable=warning,performance,portability" "--force" "--inline-suppr" "--quiet")

    if(NOT DEFINED STDGPU_COMPILE_WARNING_AS_ERROR)
        message(FATAL_ERROR "STDGPU_COMPILE_WARNING_AS_ERROR not defined.")
    endif()

    if(STDGPU_COMPILE_WARNING_AS_ERROR)
        list(APPEND ${STDGPU_OUTPUT_PROPERTY_CPPCHECK} "--error-exitcode=1")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_PROPERTY_CPPCHECK} ${${STDGPU_OUTPUT_PROPERTY_CPPCHECK}} PARENT_SCOPE)
endfunction()
