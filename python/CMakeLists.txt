# Collect all source files from the current directory for the Python module 
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Set python version to 2.7 by default, if not specified
if(NOT PYBIND11_PYTHON_VERSION)
    set(PYBIND11_PYTHON_VERSION 2.7)
endif()

# Include the Eigen include path for pybind11
include_directories(${EXTERNAL_INCLUDE_PATH}/eigenx)

# Find a local installation of pybind11
find_package(pybind11 REQUIRED
    NO_DEFAULT_PATH  # used to prevent system installed pybind11
    PATHS ${EXTERNAL_BUILD_PATH})

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)

# Create an install target for the python module
install(TARGETS optima DESTINATION "lib" COMPONENT libraries)