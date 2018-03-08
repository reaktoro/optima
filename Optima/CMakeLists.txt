# Collect all header files from the current directory for the C++ library
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

# Collect all source files from the current directory for the C++ library
file(GLOB_RECURSE CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# The name of the shared and static libraries
set(OPTIMA_SHARED_LIB ${PROJECT_NAME}${SUFFIX_SHARED_LIBS})
set(OPTIMA_STATIC_LIB ${PROJECT_NAME}${SUFFIX_STATIC_LIBS})

# Compile Optima cpp files into object files
add_library(OptimaObject OBJECT ${HPP_FILES} ${CPP_FILES})

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
