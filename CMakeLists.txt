# Set cmake version requirement
cmake_minimum_required(VERSION 3.0)

# Set the name of the project
project(Optima VERSION 0.1 LANGUAGES CXX)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Include this file to output colored text messages
include(cmake/ColoredText.cmake)

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

# Set OPTIMA_INSTALL_PREFIX to /cmake-install-prefix-path/Optima/v1.2.3 if not defined 
if(NOT DEFINED OPTIMA_INSTALL_PREFIX)
    set(OPTIMA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/v${PROJECT_VERSION})
endif()

# Set the output directories of the built libraries and binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Set the directories where libraries and header files are installed
set(OPTIMA_INSTALL_LIBDIR     ${OPTIMA_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(OPTIMA_INSTALL_BINDIR     ${OPTIMA_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
set(OPTIMA_INSTALL_INCLUDEDIR ${OPTIMA_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

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

# The default path where the external dependencies are installed 
set(OPTIMA_EXTERNAL_BUILD_PATH_DEFAULT ${CMAKE_SOURCE_DIR}/build/dependencies)

# Set the build path for the external dependencies, if not provided, and install them there
if(NOT DEFINED OPTIMA_EXTERNAL_BUILD_PATH)
    # Check if the dependencies have been installed in the default build/dependencies dir 
    if(EXISTS ${OPTIMA_EXTERNAL_BUILD_PATH_DEFAULT}/include)
        set(OPTIMA_EXTERNAL_BUILD_PATH ${OPTIMA_EXTERNAL_BUILD_PATH_DEFAULT})
    else()
    message(FATAL_ERROR "\nCould not find the installed external dependencies. "
        "Have you already installed it using:\n"
        "\tcmake -P dependencies/install\n"
        "issued from the root directory? "
        "If you installed the dependencies in a directory different than the default:\n"
        "\t${OPTIMA_EXTERNAL_BUILD_PATH_DEFAULT}\n"
        "you need to set OPTIMA_EXTERNAL_BUILD_PATH variable to that path using:\n"
        "\tcmake . -DOPTIMA_EXTERNAL_BUILD_PATH=/some/to/installed/dependencies")
    endif()
endif()

# Set path variables for the include and lib directories of the installed dependencies
set(OPTIMA_EXTERNAL_INCLUDE_PATH ${OPTIMA_EXTERNAL_BUILD_PATH}/include)
set(OPTIMA_EXTERNAL_LIBRARY_PATH ${OPTIMA_EXTERNAL_BUILD_PATH}/lib)

# Add the root path of this project to the cmake include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add the include path of the installed external dependencies to the cmake include directories
include_directories(${OPTIMA_EXTERNAL_INCLUDE_PATH})

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

# Print a summary of the configuration
#if(NOT DEFINED OPTIMA_DISABLE_SUMMARY)
#    message("====================================================================================================")
#    message("Summary")
#    message("====================================================================================================")
#    message("${PROJECT_NAME} will be compiled in ${CMAKE_BUILD_TYPE} mode.")
#    message("${PROJECT_NAME} will be installed in:")
#    message("    ${OPTIMA_INSTALL_PREFIX}")
#    message("----------------------------------------------------------------------------------------------------")
#    message("*** To disable this summary, set OPTIMA_DISABLE_SUMMARY to ON ***")
#    message("====================================================================================================")
#endif()
