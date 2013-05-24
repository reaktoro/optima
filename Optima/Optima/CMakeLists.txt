# Recursively collect all the header files in the current directory
file(GLOB_RECURSE HEADER_FILES *.hpp)

# Recursively collect all the source files in the current directory
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# Create the library using the collected source files
add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

# Install the target library
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# Install the header files preserving directory hierarchy
install(DIRECTORY ${OPTIMA_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp")
