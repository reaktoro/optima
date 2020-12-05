# Set cmake version requirement
cmake_minimum_required(VERSION 3.13)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Use ccache to speed up repeated compilations
include(CCache)

# Set the name of the project
project(Optima VERSION 0.1 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Check if a conda environment is active
include(CondaAware)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Checking CMAKE_INSTALL_PREFIX
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# Define which Optima targets to build
option(OPTIMA_BUILD_DEMOS  "Build demos." ON)
option(OPTIMA_BUILD_PYTHON "Build the python wrappers." ON)
option(OPTIMA_BUILD_DOCS   "Build documentation." OFF)
option(OPTIMA_BUILD_BENCH  "Build benchmarks." OFF)
option(OPTIMA_BUILD_ALL    "Build everything." OFF)

# Define if shared library should be build instead of static.
option(BUILD_SHARED_LIBS "Build shared libraries." ON)

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

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    # The build type selection for the project
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type for ${PROJECT_NAME}." FORCE)

    # The build type options for the project
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

# Build the C++ library Optima
add_subdirectory(Optima)

# Build the python wrappers
if(OPTIMA_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    find_package (Python COMPONENTS Interpreter Development)
    message(STATUS "Python library directory: ${Python_SITEARCH}")
    add_subdirectory(python)
endif()

# Build the demonstration applications
if(OPTIMA_BUILD_DEMOS)
    add_subdirectory(demos)
endif()

# Build the project documentation
if(OPTIMA_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Add target "tests" for manual execution of tests, as `make tests`
add_custom_target(tests
    COMMAND pytest
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests")

# Install the cmake config files that permit users to use find_package(Optima)
include(OptimaInstallCMakeConfigFiles)
