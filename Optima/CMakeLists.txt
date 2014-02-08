# Specify the name of the project
project(Optima)

# Specify the minimum required version of cmake for the project
cmake_minimum_required(VERSION 2.8)

# Set some auxiliary variables
set(OPTIMA_DEMOS_DIR     ${CMAKE_SOURCE_DIR}/demos)
set(OPTIMA_RESOURCES_DIR ${CMAKE_SOURCE_DIR}/resources)
set(OPTIMA_SOURCE_DIR    ${CMAKE_SOURCE_DIR}/Optima)
set(OPTIMA_TESTS_DIR     ${CMAKE_SOURCE_DIR}/tests)
set(OPTIMA_UTILITIES_DIR ${CMAKE_SOURCE_DIR}/utilities)
set(OPTIMA_RUNTIME_DIR   ${CMAKE_BINARY_DIR}/bin)
set(OPTIMA_LIBRARY_DIR   ${CMAKE_BINARY_DIR}/lib)
set(OPTIMA_ARQUIVE_DIR   ${CMAKE_BINARY_DIR}/lib)

# Make sure the compilation supports c++11 features
add_definitions(-std=c++11 -fmessage-length=0 -fPIC)

# Set the default build type to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Set the output path of the executable binaries produced by the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OPTIMA_RUNTIME_DIR}"
    CACHE STRING "Binary build output directory")

# Set the output path of the shared library binaries produced by the project
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OPTIMA_LIBRARY_DIR}"
    CACHE STRING "Shared library build output directory")

# Set the output path of the static library binaries produced by the project
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OPTIMA_ARQUIVE_DIR}"
    CACHE STRING "Static library build output directory")

# Set the suffix of the debug libraries
set(CMAKE_DEBUG_POSTFIX "-d")

# Specify the include directories for the project
include_directories(${CMAKE_SOURCE_DIR})

# Find the required libraries
find_library(OPTIMA_LIB Optima)

# Add the following directories to be built
add_subdirectory(Optima)
add_subdirectory(demos)
add_subdirectory(tests)

enable_testing()

file(GLOB_RECURSE TEST_FILES ${CMAKE_BINARY_DIR}/bin/tests/*)
list(REVERSE TEST_FILES)

foreach(testfile ${TEST_FILES})
    get_filename_component(testpath ${testfile} PATH)
    get_filename_component(testname ${testfile} NAME_WE)
    add_test(${testname} ${testpath}/${testname})
endforeach()
