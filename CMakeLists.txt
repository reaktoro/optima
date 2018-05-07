# Set cmake version requirement
cmake_minimum_required(VERSION 3.1)

# Set the name of the project
project(Optima VERSION 0.1 LANGUAGES CXX)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Define which Optima targets to build
option(OPTIMA_BUILD_ALL    "Build everything." OFF)
option(OPTIMA_BUILD_DEMOS  "Build demos." OFF)
option(OPTIMA_BUILD_DOCS   "Build documentation." OFF)
option(OPTIMA_BUILD_PYTHON "Build the python wrappers." ON)
option(OPTIMA_BUILD_BENCH  "Build benchmarks." OFF)

# Option to allow or not Eigen to allocate memory at runtime
option(EIGEN_RUNTIME_NO_MALLOC "Allow or not Eigen to allocate memory at runtime" OFF)

# Define EIGEN_RUNTIME_NO_MALLOC if Eigen is not allowed to allocate memmory at runtime
if(EIGEN_RUNTIME_NO_MALLOC)
    add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif()

# Modify the BUILD_XXX variables accordingly to OPTIMA_BUILD_ALL
if(OPTIMA_BUILD_ALL MATCHES ON)
    set(OPTIMA_BUILD_DEMOS  ON)
    set(OPTIMA_BUILD_DOCS   ON)
    set(OPTIMA_BUILD_PYTHON ON)
endif()

# Set the output directories of the built libraries and binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Define which types of libraries to build
option(BUILD_SHARED_LIBS "Build shared libraries." ON)
option(BUILD_STATIC_LIBS "Build static libraries." ON)

# Optima currently is not setup to produce a dynamic library using MSVC, only static
if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(BUILD_SHARED_LIBS OFF)
endif()

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    # The build type selection for the project
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type for ${PROJECT_NAME}." FORCE)

    # The build type options for the project
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

# Check if the DEPENDENCIES_PATH cmake variable was provided as argument
if(DEFINED DEPENDENCIES_PATH)
    set(OPTIMA_DEPENDENCIES_PATH ${DEPENDENCIES_PATH})
endif()

# Build and install the dependencies if the path to the installed dependencies is 
# not provided via the variable OPTIMA_DEPENDENCIES_PATH
if(NOT DEFINED OPTIMA_DEPENDENCIES_PATH)
    # Define the OPTIMA_DEPENDENCIES_PATH to default path under build/dependencies
    set(OPTIMA_DEPENDENCIES_PATH ${CMAKE_SOURCE_DIR}/build/dependencies 
        CACHE PATH "Path where the library dependencies of Optima were installed.")
    
    # Execute the installation script for the library dependencies
    execute_process(COMMAND ${CMAKE_COMMAND} -P dependencies/install
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Set path variables for the include and lib directories of the installed dependencies
set(OPTIMA_DEPENDENCIES_INCLUDE_PATH ${OPTIMA_DEPENDENCIES_PATH}/${CMAKE_INSTALL_INCLUDEDIR})
set(OPTIMA_DEPENDENCIES_LIBRARY_PATH ${OPTIMA_DEPENDENCIES_PATH}/${CMAKE_INSTALL_LIBDIR})

# Add the root path of this project to the cmake include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add the include path of the installed external dependencies to the cmake include directories
include_directories(${OPTIMA_DEPENDENCIES_INCLUDE_PATH})

# Build the C++ library Optima
add_subdirectory(Optima)

# Build the python wrappers
if(OPTIMA_BUILD_PYTHON)
    add_subdirectory(python)
else()
    add_subdirectory(python EXCLUDE_FROM_ALL)
endif()

# Build the demonstration applications
if(OPTIMA_BUILD_DEMOS)
    add_subdirectory(demos)
else()
    add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

# Build the project documentation
if(OPTIMA_BUILD_DOCS)
    add_subdirectory(docs)
else()
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

# Add target "dependencies" for manual building of the external dependencies, as `make dependencies`
add_custom_target(dependencies
    COMMAND ${CMAKE_COMMAND} -P dependencies/install
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Add target "demos" for manual building of demos, as `make demos`, if OPTIMA_BUILD_DEMOS is OFF
add_custom_target(demos
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/demos")

# Add target "tests" for manual execution of tests, as `make tests`
add_custom_target(tests
    COMMAND pytest
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests")
