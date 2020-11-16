# Collect all header files from the current directory for the C++ library
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

# Collect all source files from the current directory for the C++ library
file(GLOB_RECURSE CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# Compile Optima cpp files into a library
add_library(Optima ${HPP_FILES} ${CPP_FILES})

# Add an alias Optima::Optima to the target library Optima
add_library(Optima::Optima ALIAS Optima)

# Ensure Optima is compiled with c++17 features and propagate this to dependent codes
target_compile_features(Optima PUBLIC cxx_std_17)

# Set some target properties
set_target_properties(Optima PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF)

# Add the root directory of the project to the include list
target_include_directories(Optima PRIVATE ${PROJECT_SOURCE_DIR})

# Add the include paths to the Optima target and propagate to dependent codes
target_include_directories(Optima
    PRIVATE ${PROJECT_SOURCE_DIR}
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#==============================================================================
#
# PYTHON BINDINGS COMPILATION CONFIGURATION
#
#==============================================================================

if(OPTIMA_BUILD_PYTHON)

    # Include the Eigen include path for pybind11
    include_directories(${PROJECT_SOURCE_DIR}/Optima/deps/eigen3)

    # Ensure pybind11 uses C++17 standard
    set(PYBIND11_CPP_STANDARD -std=c++17)

    # Find pybind11, needed to compile the python bindings
    find_package(pybind11 REQUIRED)

    # Recursively collect all python binding files from the current directory
    file(GLOB_RECURSE PY_CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py.cxx)

    # Compile Optima *.py.cxx files into a Python module
    pybind11_add_module(optima4py ${PY_CXX_FILES})

    # Add the root directory of the project to the include list
    target_include_directories(optima4py PRIVATE ${PROJECT_SOURCE_DIR})

    # Link optima4py against Optima C++ library
    target_link_libraries(optima4py LINK_PRIVATE Optima::Optima)

    # This is needed to avoid compilation issues when using clang-9 and c++17
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10.0.0)
        target_compile_options(optima PRIVATE -fsized-deallocation)
    endif()

endif()

#==============================================================================
#
# INSTALL CONFIGURATION
#
#==============================================================================

# Install Optima C++ library
install(TARGETS Optima
    EXPORT OptimaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries)

# Install Optima header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Optima
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers
    FILES_MATCHING PATTERN "*.hpp")

# Install deps header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Optima/deps
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Optima COMPONENT headers)
