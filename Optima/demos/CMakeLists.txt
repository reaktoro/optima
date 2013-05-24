# Specify the name of the current project
project(demos)

# Specify the minimum required version of cmake for the project
cmake_minimum_required(VERSION 2.8)

# Update the output path of the executable binaries produced by this project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME})

# Recursively collect all the header files in the current directory
file(GLOB_RECURSE HEADER_FILES *.hpp)

# Recursively collect all the source files in the current directory
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# Create an executable for each source file within the current directory
foreach(sourcefile ${SOURCE_FILES})
    get_filename_component(filename ${sourcefile} NAME_WE)
    add_executable(${filename} ${sourcefile})
    target_link_libraries(${filename} ${CMAKE_PROJECT_NAME})
endforeach()
