# Specify the include directories for the project
include_directories(${OPTIMA_SOURCE_DIR})

# Recursively collect all the header files in the current directory
file(GLOB_RECURSE HEADER_FILES *.hpp)

# Recursively collect all the source files in the current directory
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# Create the library using the collected source files
add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
