# Include the Eigen include path for pybind11
include_directories(${PROJECT_SOURCE_DIR}/Optima/deps/eigen3)

# Find pybind11, needed to compile the python bindings
find_package(pybind11 REQUIRED)

# Collect all source files from the current directory for the Python module
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)

# Create an install target for the python module
install(TARGETS optima
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
