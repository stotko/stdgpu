
find_package(Doxygen 1.9.6 EXACT QUIET)

if(NOT Doxygen_FOUND)
    include(ExternalProject)

    ExternalProject_Add(
        doxygen
        PREFIX doxygen
        URL https://github.com/doxygen/doxygen/archive/refs/tags/Release_1_9_6.zip
        URL_HASH SHA256=28ab2c5d2ba1d3ed56c538125f1cad7ca8a09992e8b3892e9533736268ff6426
        DOWNLOAD_DIR "${STDGPU_EXTERNAL_DIR}/doxygen"
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
        EXCLUDE_FROM_ALL TRUE
    )

    ExternalProject_Get_Property(doxygen BINARY_DIR)
    add_executable(Doxygen::doxygen IMPORTED)
    set_target_properties(Doxygen::doxygen PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/bin/doxygen")

    add_dependencies(Doxygen::doxygen doxygen)
endif()


# The Doxygen Awesome has to be modified due to two major reasons:
# - Doxysphinx is used in a later stage and puts the doxygen parts into a .doxygen-content container
# - The dark mode switch and the respective variables of the sphinx-book-theme will be used
#
# Therefore, use the modified stylesheet from ROCm released under the MIT license:
# https://github.com/RadeonOpenCompute/rocm-docs-core/blob/develop/src/rocm_docs/data/_doxygen/extra_stylesheet.css


set(STDGPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
get_filename_component(STDGPU_INCLUDE_DIR ${STDGPU_INCLUDE_DIR} ABSOLUTE)


if(STDGPU_COMPILE_WARNING_AS_ERROR)
    set(STDGPU_DOXYGEN_WARN_AS_ERROR YES)
else()
    set(STDGPU_DOXYGEN_WARN_AS_ERROR NO)
endif()


set(STDGPU_DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(STDGPU_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file(${STDGPU_DOXYFILE_IN}
               ${STDGPU_DOXYFILE}
               @ONLY)


find_package(Python COMPONENTS Interpreter REQUIRED)

find_program(STDGPU_DOXYSPHINX doxysphinx REQUIRED)
find_program(STDGPU_SPHINX sphinx-build REQUIRED)


set(STDGPU_SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")
set(STDGPU_SPHINX_DOXYGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doxygen")
set(STDGPU_SPHINX_FORCE_BUILDING "-E")
set(STDGPU_SPHINX_PARALLEL "-j" "auto")

add_custom_target(stdgpu_doc
                 # Pre. Clean previous build
                 COMMAND ${CMAKE_COMMAND} -E rm -Rf "${STDGPU_SPHINX_OUTPUT_DIR}" "${STDGPU_SPHINX_DOXYGEN_DIR}"
                 # 1. Run Doxygen
                 COMMAND Doxygen::doxygen "${STDGPU_DOXYFILE}"
                 # 2. Run Doxysphinx
                 COMMAND ${STDGPU_DOXYSPHINX} build "${CMAKE_CURRENT_SOURCE_DIR}" "${STDGPU_SPHINX_OUTPUT_DIR}" "${STDGPU_SPHINX_DOXYGEN_DIR}"
                 # 2.1. Remove unused main page
                 COMMAND ${CMAKE_COMMAND} -E rm "${STDGPU_SPHINX_DOXYGEN_DIR}/index.rst"
                 # 2.2. Fix Doxysphinx files
                 COMMAND Python::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/fix_html_titles"
                 # 3. Run Sphinx
                 COMMAND ${STDGPU_SPHINX} "-b" "html" ${STDGPU_SPHINX_FORCE_BUILDING} ${STDGPU_SPHINX_PARALLEL} "${CMAKE_CURRENT_SOURCE_DIR}" "${STDGPU_SPHINX_OUTPUT_DIR}"
                 COMMENT "Generating stdgpu documentation"
                 VERBATIM)
