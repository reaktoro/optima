# Collect all source files from the current directory for the Python module 
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# The include path for Eigen library compatible with pybind11
set(EIGEN_INCLUDE_PATH_FOR_PYBIND11 ${CMAKE_SOURCE_DIR}/external/include/${PROJECT_NAME}/external/eigenx)

# Include the Eigen include path for pybind11
include_directories(${EIGEN_INCLUDE_PATH_FOR_PYBIND11})

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)
