# Require a certain version of cmake
cmake_minimum_required(VERSION 3.0)

# Recursively collect all header files from the current directory
file(GLOB_RECURSE HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp *.hxx)

# Recursively collect all source files from the current directory
file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# The name of the shared and static libraries
set(OPTIMA_SHARED_LIB ${PROJECT_NAME}${SUFFIX_SHARED_LIBS})
set(OPTIMA_STATIC_LIB ${PROJECT_NAME}${SUFFIX_STATIC_LIBS})

# Set the list of names of the third-party targets and libraries
# Compile Optima into object files
add_library(OptimaObject OBJECT ${HEADER_FILES} ${SOURCE_FILES})

# Let cotire also consider the Eigen library in its attempt to speedup compilation times
#set_target_properties(OptimaObject PROPERTIES
#   COTIRE_PREFIX_HEADER_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/Optima/Math/Eigen")
#
# Apply cotire to the library target
#cotire(OptimaObject)

# Check if a shared library for Optima must be built
if(BUILD_SHARED_LIBS)
    add_library(OptimaShared SHARED $<TARGET_OBJECTS:OptimaObject>)
    set_target_properties(OptimaShared PROPERTIES OUTPUT_NAME Optima)
    install(TARGETS OptimaShared DESTINATION "lib" COMPONENT libraries)
endif()

# Check if a static library for Optima must be built
if(BUILD_STATIC_LIBS)
    add_library(OptimaStatic STATIC $<TARGET_OBJECTS:OptimaObject>)
    set_target_properties(OptimaStatic PROPERTIES OUTPUT_NAME Optima)
    install(TARGETS OptimaStatic DESTINATION "lib" COMPONENT libraries)
endif()

# Create an install target for Optima header files
install(DIRECTORY ${OPTIMA_SOURCE_DIR}
    DESTINATION "include" COMPONENT headers
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx")

# Create an install target for Eigen header files
install(DIRECTORY ${OPTIMA_SOURCE_DIR}/Eigen
    DESTINATION "include/Optima" COMPONENT headers)
