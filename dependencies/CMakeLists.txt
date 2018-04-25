# Set cmake version requirement
cmake_minimum_required(VERSION 3.0)

# Set the name of the project
project(dependencies)

# Include the ExternalProject module
include(ExternalProject)

# Set the default build type to Release for the external dependencies
if(NOT CMAKE_BUILD_TYPE)
    # The build type selection for the external dependencies
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the common build type for the external dependencies." FORCE)

    # The build type options for the external dependencies
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

# Output a message detailing the build type used
message(STATUS "Using build type `${CMAKE_BUILD_TYPE}` for all external dependencies.")

# Set the path where the external dependencies should be installed
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

# Set the common cmake arguments for all external dependencies
set(COMMON_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

# Update, and download if needed, the external dependencies in the form of git submodule
execute_process(COMMAND git submodule update --init --recursive)

# Build and install the Eigenx library
ExternalProject_Add(EIGENX
    PREFIX build
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigenx
    CMAKE_ARGS ${COMMON_CMAKE_ARGS}
    CMAKE_ARGS -DBUILD_TESTING=OFF)

# Build the pybind11 library and do an intermediate install to EXTERNAL_INTERMEDIATE_INSTALL_DIR
ExternalProject_Add(PYBIND11
    PREFIX build
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11
    CMAKE_ARGS ${COMMON_CMAKE_ARGS}
    CMAKE_ARGS -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF)

# Create target eigenx to force the build of the EIGENX dependency by removing its timestamp file
add_custom_target(eigenx
    COMMAND ${CMAKE_COMMAND} -E remove build/src/EIGENX-stamp/EIGENX-build
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

# Create target pybind11 to force the build of the PYBIND11 dependency by removing its timestamp file
add_custom_target(pybind11
    COMMAND ${CMAKE_COMMAND} -E remove build/src/PYBIND11-stamp/PYBIND11-build
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
