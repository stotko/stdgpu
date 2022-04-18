
find_program(CLANG_TIDY_EXECUTABLE
             NAMES
             "clang-tidy")

if(CLANG_TIDY_EXECUTABLE)
    execute_process(COMMAND "${CLANG_TIDY_EXECUTABLE}" "--version" OUTPUT_VARIABLE CLANG_TIDY_VERSION_TEXT)
    string(REGEX MATCH "LLVM version ([^\n]*)" CLANG_TIDY_VERSION_TEXT_CUT "${CLANG_TIDY_VERSION_TEXT}")
    set(CLANG_TIDY_VERSION "${CMAKE_MATCH_1}")

    unset(CLANG_TIDY_VERSION_TEXT_CUT)
    unset(CLANG_TIDY_VERSION_TEXT)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangTidy
                                  REQUIRED_VARS CLANG_TIDY_EXECUTABLE
                                  VERSION_VAR CLANG_TIDY_VERSION)

if(ClangTidy_FOUND)
    add_executable(ClangTidy::ClangTidy IMPORTED)
    set_target_properties(ClangTidy::ClangTidy PROPERTIES IMPORTED_LOCATION "${CLANG_TIDY_EXECUTABLE}")
endif()
