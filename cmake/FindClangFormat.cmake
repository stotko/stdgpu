
find_program(CLANG_FORMAT_EXECUTABLE
             NAMES
             "clang-format-10" # Prefer exact version
             "clang-format")

if(CLANG_FORMAT_EXECUTABLE)
    execute_process(COMMAND "${CLANG_FORMAT_EXECUTABLE}" "--version" OUTPUT_VARIABLE CLANG_FORMAT_VERSION_TEXT)
    string(REGEX MATCH "clang-format version ([^\n]*)" CLANG_FORMAT_VERSION_TEXT_CUT "${CLANG_FORMAT_VERSION_TEXT}")
    set(CLANG_FORMAT_VERSION "${CMAKE_MATCH_1}")

    unset(CLANG_FORMAT_VERSION_TEXT_CUT)
    unset(CLANG_FORMAT_VERSION_TEXT)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangFormat
                                  REQUIRED_VARS CLANG_FORMAT_EXECUTABLE
                                  VERSION_VAR CLANG_FORMAT_VERSION)

if(ClangFormat_FOUND)
    add_executable(ClangFormat::ClangFormat IMPORTED)
    set_target_properties(ClangFormat::ClangFormat PROPERTIES IMPORTED_LOCATION "${CLANG_FORMAT_EXECUTABLE}")
endif()
