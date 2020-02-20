
# Check include path
set(STDGPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
get_filename_component(STDGPU_INCLUDE_DIR ${STDGPU_INCLUDE_DIR} ABSOLUTE)


find_package(Doxygen 1.8.13)


if(Doxygen_FOUND)
    set(STDGPU_HAVE_DOXYGEN ON CACHE BOOL "Support for documentation generation using Doxygen" FORCE)

    if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
        set(STDGPU_DOXYGEN_WARN_AS_ERROR YES)
    else()
        set(STDGPU_DOXYGEN_WARN_AS_ERROR NO)
    endif()

    set(STDGPU_DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
    set(STDGPU_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
    set(STDGPU_DOC_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    configure_file(${STDGPU_DOXYFILE_IN}
                   ${STDGPU_DOXYFILE}
                   @ONLY)

    add_custom_target(stdgpu_doc
                    COMMAND ${DOXYGEN_EXECUTABLE} ${STDGPU_DOXYFILE}
                    WORKING_DIRECTORY ${STDGPU_DOC_BUILD_DIR}
                    COMMENT "Generating stdgpu API documentation with Doxygen"
                    VERBATIM)


#    install(DIRECTORY "${STDGPU_DOC_BUILD_DIR}/doxygen-html/"
#            DESTINATION ${STDGPU_DOC_INSTALL_DIR}
#            COMPONENT stdgpu)

endif()
