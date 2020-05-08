# Specify the minimum cmake version
cmake_minimum_required(VERSION 3.13)

# Specify the name of your project
project(app)

# Find the Optima dependency.
find_package(Optima REQUIRED)

# Create an executable named app with source code main.cpp
add_executable(app main.cpp)

# Link your application against Optima C++ library.
target_link_libraries(app Optima::Optima)
