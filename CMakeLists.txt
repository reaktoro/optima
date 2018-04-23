# Set cmake version requirement
cmake_minimum_required(VERSION 3.0)

# Set the name of the project
project(Optima)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Set the version of the project
set(OPTIMA_VERSION_MAJOR "0")
set(OPTIMA_VERSION_MINOR "1")
set(OPTIMA_VERSION_MICRO "0")
set(OPTIMA_VERSION "${OPTIMA_VERSION_MAJOR}.${OPTIMA_VERSION_MINOR}.${OPTIMA_VERSION_MICRO}")

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
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define which types of libraries to build
option(BUILD_SHARED_LIBS "Build shared libraries." ON)
option(BUILD_STATIC_LIBS "Build static libraries." ON)

# Set libraries to be compiled with position independent code mode (i.e., fPIC option in GNU compilers)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the list of compiler flags for GNU compiler
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options(-std=c++11 -Wall)
endif()

# Set the list of compiler flags for MSVC compiler
if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_compile_options("/W0 -D_SCL_SECURE_NO_WARNINGS /MP4")
endif()

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

# Output a message detailing the build type used
message(STATUS "Using build type `${CMAKE_BUILD_TYPE}` for ${PROJECT_NAME}.")

# Set the build path for the external dependencies, if not provided
if(NOT DEFINED EXTERNAL_BUILD_PATH)
    set(EXTERNAL_BUILD_PATH ${CMAKE_SOURCE_DIR}/external)
endif()

# Set path variables for the include and lib directories of the installed dependencies
set(EXTERNAL_INCLUDE_PATH ${EXTERNAL_BUILD_PATH}/include)
set(EXTERNAL_LIBRARY_PATH ${EXTERNAL_BUILD_PATH}/lib)

# Copy Eigen header files to PROJECT_NAME/external/eigenx
file(COPY ${EXTERNAL_INCLUDE_PATH}/eigenx
    DESTINATION ${EXTERNAL_INCLUDE_PATH}/${PROJECT_NAME}/external/)

# Add the root path of this project to the cmake include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add the include path of the installed external dependencies to the cmake include directories
include_directories(${EXTERNAL_INCLUDE_PATH})

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
