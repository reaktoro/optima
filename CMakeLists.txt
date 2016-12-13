# Set cmake version requirement
cmake_minimum_required(VERSION 3.0)

# Set the name of the project
project(Optima)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include this module to enable testing with CTest and dashboard submissions to CDash
include(CTest)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Set the version of the project
set(OPTIMA_VERSION_MAJOR "0")
set(OPTIMA_VERSION_MINOR "1")
set(OPTIMA_VERSION_MICRO "0")
set(OPTIMA_VERSION "${OPTIMA_VERSION_MAJOR}.${OPTIMA_VERSION_MINOR}.${OPTIMA_VERSION_MICRO}")

# Define which components of Optima to build
option(BUILD_ALL         "Build everything." OFF)
option(BUILD_DEMOS       "Build demos." OFF)
option(BUILD_DOCS        "Build documentation." OFF)
option(BUILD_PYTHON      "Build the python wrappers and python package reaktoro." OFF)
option(BUILD_TESTS       "Build tests." OFF)

# Set the CTest option BUILD_TESTING to the value of BUILD_TESTS
set(BUILD_TESTING BUILD_TEST)

# Modify the BUILD_XXX variables accordingly to BUILD_ALL
if(BUILD_ALL MATCHES ON)
    set(BUILD_DEMOS       ON)
    set(BUILD_DOCS        ON)
    set(BUILD_PYTHON      ON)
    set(BUILD_TESTS       ON)
endif()


# Find python interpreter if needed
if(BUILD_PYTHON)
    find_package(PythonInterp)
    if(NOT PYTHONINTERP_FOUND)
        set(BUILD_PYTHON OFF)
        message(WARNING "Python executable was not found. "
            "Cannot build python wrappers and packages for Optima.")
    endif()
endif()

# Set the output directories of the built libraries and binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define which types of libraries to build
option(BUILD_SHARED_LIBS "Build shared libraries." ON)
option(BUILD_STATIC_LIBS "Build static libraries." ON)

# Optima currently is not setup to produce a dynamic library using MSVC, only static
if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(BUILD_SHARED_LIBS OFF)
endif()

# Define custom options
option(ENABLE_TESTING "Enable testing." on)

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set libraries to be compiled with position independent code mode (i.e., fPIC option in GNU compilers)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the list of compiler flags for GNU compiler
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options(-std=c++11 -Wall -Wno-ignored-attributes -Wno-pedantic -Wno-variadic-macros -Wno-deprecated)
#    add_compile_options(-std=c++11 -Wall -Wno-misleading-indentation -Wno-ignored-attributes -Wno-pedantic -Wno-variadic-macros -Wno-deprecated)
endif()

# Set the list of compiler flags for MSVC compiler
if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_compile_options("/W0 -D_SCL_SECURE_NO_WARNINGS /MP4")
endif()

# Set the project directory names
set(OPTIMA_DEMOS_DIR      ${CMAKE_SOURCE_DIR}/demos)
set(OPTIMA_DOCS_DIR       ${CMAKE_SOURCE_DIR}/docs)
set(OPTIMA_RESOURCES_DIR  ${CMAKE_SOURCE_DIR}/resources)
set(OPTIMA_SOURCE_DIR     ${CMAKE_SOURCE_DIR}/Optima)

# Set the include directories
include_directories(${CMAKE_SOURCE_DIR})

# Build the C++ library Optima
add_subdirectory(Optima)

# Build the Python extension module PyOptima and the Python packages reaktoro
if(BUILD_PYTHON)
    add_subdirectory(PyOptima)
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

# Build the tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
else()
    enable_testing()
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()

# Add target "demos" for manual building of demos, as `make demos`, if BUILD_DEMOS is OFF
add_custom_target(demos
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/demos")

# Add target "tests" for manual building of tests, as `make tests`, if BUILD_TESTS is OFF
add_custom_target(tests
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
