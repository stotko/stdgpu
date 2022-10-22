function(stdgpu_print_configuration_summary)
    message(STATUS "")
    message(STATUS "************************ stdgpu Configuration Summary *************************")
    message(STATUS "")

    message(STATUS "General:")
    message(STATUS "  Version                                   :   ${stdgpu_VERSION}")
    message(STATUS "  System                                    :   ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Build type                                :   ${CMAKE_BUILD_TYPE}")

    message(STATUS "")

    message(STATUS "Build:")
    message(STATUS "  STDGPU_BACKEND                            :   ${STDGPU_BACKEND}")
    message(STATUS "  STDGPU_BUILD_SHARED_LIBS                  :   ${STDGPU_BUILD_SHARED_LIBS}")
    message(STATUS "  STDGPU_SETUP_COMPILER_FLAGS               :   ${STDGPU_SETUP_COMPILER_FLAGS}")
    message(STATUS "  STDGPU_COMPILE_WARNING_AS_ERROR           :   ${STDGPU_COMPILE_WARNING_AS_ERROR}")
    message(STATUS "  STDGPU_ANALYZE_WITH_CLANG_TIDY            :   ${STDGPU_ANALYZE_WITH_CLANG_TIDY}")
    message(STATUS "  STDGPU_ANALYZE_WITH_CPPCHECK              :   ${STDGPU_ANALYZE_WITH_CPPCHECK}")

    message(STATUS "")

    message(STATUS "Configuration:")
    message(STATUS "  STDGPU_ENABLE_CONTRACT_CHECKS             :   ${STDGPU_ENABLE_CONTRACT_CHECKS}")
    message(STATUS "  STDGPU_USE_32_BIT_INDEX                   :   ${STDGPU_USE_32_BIT_INDEX}")

    message(STATUS "")

    message(STATUS "Examples:")
    message(STATUS "  STDGPU_BUILD_EXAMPLES                     :   ${STDGPU_BUILD_EXAMPLES}")

    message(STATUS "")

    message(STATUS "Benchmarks:")
    message(STATUS "  STDGPU_BUILD_BENCHMARKS                   :   ${STDGPU_BUILD_BENCHMARKS}")

    message(STATUS "")

    message(STATUS "Tests:")
    message(STATUS "  STDGPU_BUILD_TESTS                        :   ${STDGPU_BUILD_TESTS}")
    message(STATUS "  STDGPU_BUILD_TEST_COVERAGE                :   ${STDGPU_BUILD_TEST_COVERAGE}")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        if(DEFINED CMAKE_VERIFY_INTERFACE_HEADER_SETS)
            message(STATUS "  CMAKE_VERIFY_INTERFACE_HEADER_SETS        :   ${CMAKE_VERIFY_INTERFACE_HEADER_SETS}")
        else()
            message(STATUS "  CMAKE_VERIFY_INTERFACE_HEADER_SETS        :   <Not Defined> (-> OFF)")
        endif()
    endif()

    message(STATUS "")

    message(STATUS "Documentation:")
    if(STDGPU_HAVE_DOXYGEN)
        message(STATUS "  Doxygen                                   :   YES")
    else()
        message(STATUS "  Doxygen                                   :   NO")
    endif()

    message(STATUS "")
    message(STATUS "*******************************************************************************")
    message(STATUS "")
endfunction()
