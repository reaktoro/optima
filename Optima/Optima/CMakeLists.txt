# Recursively collect all the header files in the current directory
file(GLOB_RECURSE HEADER_FILES *.hpp)

# Recursively collect all the source files in the current directory
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# The names of the shared and static libraries
set(OPTIMA_SHARED_LIB ${PROJECT_NAME})
set(OPTIMA_STATIC_LIB ${PROJECT_NAME}-static)

# Create the library using the collected source files
add_library(${OPTIMA_SHARED_LIB} SHARED ${HEADER_FILES} ${SOURCE_FILES})
add_library(${OPTIMA_STATIC_LIB} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# Install the target library
install(TARGETS ${OPTIMA_SHARED_LIB} ${OPTIMA_STATIC_LIB} DESTINATION lib)

# Install the header files preserving directory hierarchy
install(DIRECTORY ${OPTIMA_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp")
