
find_program(CPPCHECK_EXECUTABLE
             NAMES
             "cppcheck")

if(CPPCHECK_EXECUTABLE)
    execute_process(COMMAND "${CPPCHECK_EXECUTABLE}" "--version" OUTPUT_VARIABLE CPPCHECK_VERSION_TEXT)
    string(REGEX MATCH "^Cppcheck ([^\n]*)" CPPCHECK_VERSION_TEXT_CUT "${CPPCHECK_VERSION_TEXT}")
    set(CPPCHECK_VERSION "${CMAKE_MATCH_1}")

    unset(CPPCHECK_VERSION_TEXT_CUT)
    unset(CPPCHECK_VERSION_TEXT)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cppcheck
                                  REQUIRED_VARS CPPCHECK_EXECUTABLE
                                  VERSION_VAR CPPCHECK_VERSION)

if(Cppcheck_FOUND)
    add_executable(Cppcheck::Cppcheck IMPORTED)
    set_target_properties(Cppcheck::Cppcheck PROPERTIES IMPORTED_LOCATION "${CPPCHECK_EXECUTABLE}")
endif()
