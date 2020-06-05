# Include the Eigen include path for pybind11
include_directories(${PROJECT_SOURCE_DIR}/Optima/deps/eigen3)

# Find pybind11, needed to compile the python bindings
set(PYBIND11_CPP_STANDARD -std=c++17)  # Ensure pybind11 uses C++17 standard
find_package(pybind11 REQUIRED)

# Collect all source files from the current directory for the Python module
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES})

# Create an install target for the python module
install(TARGETS optima DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)

# Add the root directory of the project to the include list
target_include_directories(optima PRIVATE ${PROJECT_SOURCE_DIR})

# Link optima against Optima C++ library
target_link_libraries(optima LINK_PRIVATE Optima::Optima)

# This is needed to avoid compilation issues when using clang-9 and c++17
if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10.0.0)
    target_compile_options(optima PRIVATE -fsized-deallocation)
endif()
