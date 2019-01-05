# Set python version to 2.7 by default, if not specified
if(NOT PYBIND11_PYTHON_VERSION AND NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp REQUIRED) # PYTHON_EXECUTABLE gets set here
endif()

# Include the Eigen include path for pybind11
include_directories(${PROJECT_SOURCE_DIR}/Optima/deps/eigen3)

# Find pybind11 among the external dependencies
find_package(pybind11 REQUIRED NO_DEFAULT_PATH
    PATHS ${PROJECT_SOURCE_DIR}/deps/pybind11)

# Collect all source files from the current directory for the Python module
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)

# Create an install target for the python module
install(TARGETS optima
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)