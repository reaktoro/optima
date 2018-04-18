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
option(BUILD_ALL    "Build everything." OFF)
option(BUILD_DEMOS  "Build demos." OFF)
option(BUILD_DOCS   "Build documentation." OFF)
option(BUILD_PYTHON "Build the python wrappers." ON)
option(BUILD_BENCH  "Build benchmarks." OFF)

# Option to conveniently permit build in debug mode
option(DEBUG "Build in debug mode." OFF)

# Set debug mode if DEBUG is ON
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Option to allow or not Eigen to allocate memory at runtime
option(EIGEN_RUNTIME_NO_MALLOC "Allow or not Eigen to allocate memory at runtime" OFF)

# Define EIGEN_RUNTIME_NO_MALLOC if Eigen is not allowed to allocate memmory at runtime
if(EIGEN_RUNTIME_NO_MALLOC)
    add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif()

# Modify the BUILD_XXX variables accordingly to BUILD_ALL
if(BUILD_ALL MATCHES ON)
    set(BUILD_DEMOS  ON)
    set(BUILD_DOCS   ON)
    set(BUILD_PYTHON ON)
endif()

# # Find Eigen library and include its directory
# find_package(Eigen3 3.3.5 REQUIRED)

# # Include Eigen directory
# include_directories(${EIGEN3_INCLUDE_DIR})

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


# Build the external projects that have not been built yet
if(EXTERNAL_EXCLUDE_FROM_ALL)
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies)
endif()

# Find pybind11 library
if(BUILD_PYTHON)
    if(NOT PYBIND11_PYTHON_VERSION)
        set(PYBIND11_PYTHON_VERSION 2.7)
    endif()
    
    find_package(pybind11 2.2.2 REQUIRED)

    if(NOT pybind11_FOUND)
        set(BUILD_PYTHON OFF)
        message(WARNING "pybind11 was not found. "
            "Cannot build python wrappers and packages for Optima.")
    endif()
endif()

# Include the root directory of the project
include_directories(${CMAKE_SOURCE_DIR})

# Include directories from external projects
include_directories(${CMAKE_SOURCE_DIR}/external/include/)

# Build the C++ library Optima
add_subdirectory(Optima)

# Build the python wrappers
if(BUILD_PYTHON)
    add_subdirectory(python)
else()
    add_subdirectory(python EXCLUDE_FROM_ALL)
endif()

# Build the demonstration applications
if(BUILD_DEMOS)
    add_subdirectory(demos)
else()
    add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

# Build the project documentation
if(BUILD_DOCS)
    add_subdirectory(docs)
else()
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

# Build the benchmarks
if(BUILD_BENCH)
    add_subdirectory(bench)
else()
    add_subdirectory(bench EXCLUDE_FROM_ALL)
endif()

# Add target "demos" for manual building of demos, as `make demos`, if BUILD_DEMOS is OFF
add_custom_target(demos
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/demos")

# Add target "tests" for manual execution of tests, as `make tests`
add_custom_target(tests
    COMMAND pytest
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests")

# Add target "bench" for manual building of benchmarks, as `make bench`, if BUILD_BENCH is OFF
add_custom_target(bench
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bench")
