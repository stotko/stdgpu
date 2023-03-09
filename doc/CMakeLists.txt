
find_package(Doxygen 1.9.6 EXACT)

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


include(FetchContent)

FetchContent_Declare(
    doxygen-awesome-css
    PREFIX doxygen-awesome-css
    URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.2.0.zip
    URL_HASH SHA256=2cff61a38694895259f5fdd04599c08d86cf9fd2dafc9950bf47af09af52e227
    DOWNLOAD_DIR "${STDGPU_EXTERNAL_DIR}/doxygen-awesome-css"
)

FetchContent_GetProperties(doxygen-awesome-css)
if(NOT doxygen-awesome-css_POPULATED)
  FetchContent_Populate(doxygen-awesome-css)
endif()

set(STDGPU_DOXYGEN_AWESOME_CSS_DIR ${doxygen-awesome-css_SOURCE_DIR})


# Check include path
set(STDGPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
get_filename_component(STDGPU_INCLUDE_DIR ${STDGPU_INCLUDE_DIR} ABSOLUTE)


if(STDGPU_COMPILE_WARNING_AS_ERROR)
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
                 COMMAND Doxygen::doxygen ${STDGPU_DOXYFILE}
                 WORKING_DIRECTORY ${STDGPU_DOC_BUILD_DIR}
                 COMMENT "Generating stdgpu API documentation with Doxygen"
                 VERBATIM)


#install(DIRECTORY "${STDGPU_DOC_BUILD_DIR}/doxygen-html/"
#        DESTINATION ${STDGPU_DOC_INSTALL_DIR}
#        COMPONENT stdgpu)
