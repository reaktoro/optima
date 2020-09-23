# Include the Eigen include path for pybind11
include_directories(${PROJECT_SOURCE_DIR}/Optima/deps/eigen3)

# Collect all source files from the current directory for the Python module
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima4py ${CXX_FILES})

# Add the root directory of the project to the include list
target_include_directories(optima4py PRIVATE ${PROJECT_SOURCE_DIR})

# Link optima4py against Optima C++ library
target_link_libraries(optima4py LINK_PRIVATE Optima::Optima)

# This is needed to avoid compilation issues when using clang-9 and c++17
if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10.0.0)
    target_compile_options(optima4py PRIVATE -fsized-deallocation)
endif()
