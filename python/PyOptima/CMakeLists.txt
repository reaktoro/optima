# Include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Collect the header and source files of PyEigenx
file(GLOB_RECURSE HEADER_FILES *.hpp)
file(GLOB_RECURSE SOURCE_FILES *.cpp)

pybind11_add_module(optima PyOptima.cpp)