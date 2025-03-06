# Collect all header files from the current directory for the C++ library
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

# Collect all source files from the current directory for the C++ library
file(GLOB_RECURSE CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# Enable automatic creation of a module definition (.def) file for a SHARED library on Windows.
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# Compile Optima cpp files into a library
add_library(Optima ${HPP_FILES} ${CPP_FILES})

# Add an alias Optima::Optima to the target library Optima
add_library(Optima::Optima ALIAS Optima)

# Set some target properties
set_target_properties(Optima PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF)

# Set Optima compilation features to be propagated to client code.
target_compile_features(Optima PUBLIC cxx_std_17)

# Link Optima against its dependencies
target_link_libraries(Optima PUBLIC Eigen3::Eigen)

# Add the root directory of the project to the include list
target_include_directories(Optima PRIVATE ${PROJECT_SOURCE_DIR})

# Add the include paths to the Optima target and propagate to dependent codes
target_include_directories(Optima
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

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
