# Collect all header files from the current directory for the C++ library
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

# Collect all source files from the current directory for the C++ library
file(GLOB_RECURSE CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# The name of the shared and static libraries
set(OPTIMA_SHARED_LIB ${PROJECT_NAME}${SUFFIX_SHARED_LIBS})
set(OPTIMA_STATIC_LIB ${PROJECT_NAME}${SUFFIX_STATIC_LIBS})

# Compile Optima cpp files into object files
add_library(OptimaObject OBJECT ${HPP_FILES} ${CPP_FILES})

# Set some properties for the OptimaObject target
set_property(TARGET OptimaObject PROPERTY POSITION_INDEPENDENT_CODE ON)

# Add the root directory of the project to the include list
target_include_directories(OptimaObject PRIVATE ${PROJECT_SOURCE_DIR})

# Ensure Optima is compiled with a compiler that supports C++17
target_compile_features(OptimaObject PUBLIC cxx_std_17)

# Check if a shared library for Optima must be built
if(BUILD_SHARED_LIBS)
    add_library(OptimaShared SHARED $<TARGET_OBJECTS:OptimaObject>)
    set_target_properties(OptimaShared PROPERTIES OUTPUT_NAME Optima)
    install(TARGETS OptimaShared DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
endif()

# Check if a static library for Optima must be built
if(BUILD_STATIC_LIBS)
    add_library(OptimaStatic STATIC $<TARGET_OBJECTS:OptimaObject>)
    set_target_properties(OptimaStatic PROPERTIES OUTPUT_NAME Optima)
    install(TARGETS OptimaStatic DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
endif()

# Create an install target for Optima header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/Optima
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx")

# Create an install target for eigenx header files
install(DIRECTORY ${OPTIMA_DEPENDENCIES_INCLUDE_PATH}/eigenx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
