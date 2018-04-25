# Set python version to 2.7 by default, if not specified
if(NOT PYBIND11_PYTHON_VERSION)
    set(PYBIND11_PYTHON_VERSION 2.7)
endif()

# Include the Eigen include path for pybind11
include_directories(${OPTIMA_EXTERNAL_INCLUDE_PATH}/eigenx)

# Find pybind11 among the installed external dependencies
find_package(pybind11 REQUIRED
    NO_DEFAULT_PATH  # used to prevent system installed pybind11
    PATHS ${OPTIMA_EXTERNAL_BUILD_PATH})

# Collect all source files from the current directory for the Python module 
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)

# Create an install target for the python module
install(TARGETS optima 
    DESTINATION ${OPTIMA_INSTALL_LIBDIR} COMPONENT libraries)