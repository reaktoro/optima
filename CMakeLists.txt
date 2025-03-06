# Set cmake version requirement
cmake_minimum_required(VERSION 3.17)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Use ccache to speed up repeated compilations
include(CCache)

# Set the name of the project
project(Optima VERSION 0.6.0 LANGUAGES CXX)

# Enable parallel build if MSVC is used
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# Check if a conda environment is active
include(CondaAware)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

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

# Enable RUNPATH for executables and shared libraries on Linux for flexible library search paths
if(UNIX AND NOT APPLE)
    SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags")
    SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-new-dtags")
endif()

# Find Eigen3 either from the system, environment, or from the given directory OPTIMA_EIGEN_DIR
if(DEFINED OPTIMA_EIGEN_DIR)
    find_package(Eigen3 3.4 REQUIRED PATHS ${OPTIMA_EIGEN_DIR} NO_DEFAULT_PATH)
else()
    find_package(Eigen3 3.4 REQUIRED)
endif()
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_DIR} (found version \"${Eigen3_VERSION}\")")
endif()

# Build the C++ library Optima
add_subdirectory(Optima)

# Build the python wrappers
if(OPTIMA_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)
    find_program(PYBIND11_STUBGEN pybind11-stubgen)
    if(NOT pybind11_FOUND)
        message(WARNING "Could not find pybind11. The Python package optima will not be built!")
        set(OPTIMA_BUILD_PYTHON OFF)
    endif()
    if(NOT PYBIND11_STUBGEN)
        message(WARNING "Could not find pybind11-stubgen (available via pip or conda). There will be no stubs generated for the python package optima.")
    endif()
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

# Process sub-directory tests
add_subdirectory(tests)

# Process sub-directory scripts
add_subdirectory(scripts)

# Install the cmake config files that permit users to use find_package(Optima)
include(OptimaInstallCMakeConfigFiles)
