# Include the parent directory of PyOptima
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Find Python, Numpy and Boost.Python libraries
find_package(PythonLibs 2.7 REQUIRED)
find_package(Numpy 2.7 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

# Specify Python, Numpy and Boost include directories
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

# Collect the header and source files of PyOptima
file(GLOB_RECURSE HEADER_FILES *.hpp)
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# Suppress pedantic warnings related to ISO C++ for GNU compiler
if(${CMAKE_COMPILER_IS_GNUCXX})
    add_definitions(-Wno-pedantic -Wno-unused-local-typedefs)
endif()

# Create the PyOptima library
add_library(PyOptima SHARED ${HEADER_FILES} ${SOURCE_FILES})

# Link PyOptima against Boost.Python and Python libraries
target_link_libraries(PyOptima OptimaStatic ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# Remove the `lib` prefix of the PyOptima library file name
set_target_properties(PyOptima PROPERTIES PREFIX "" OUTPUT_NAME reaktoro)

# Check if PyOptima library is output as `.dll` (Windows) and change it to `.pyd`
if(${CMAKE_SHARED_LIBRARY_SUFFIX} MATCHES ".dll")
    set_target_properties(PyOptima PROPERTIES SUFFIX ".pyd")
endif()

# Check if PyOptima library is output as `.dylib` (MacOS) and change it to `.so`
if(${CMAKE_SHARED_LIBRARY_SUFFIX} MATCHES ".dylib")
    set_target_properties(PyOptima PROPERTIES SUFFIX ".so")
endif()

# Create an install target for PyOptima
install(TARGETS PyOptima DESTINATION "lib" COMPONENT interfaces)

# Copy the Python extension module PyOptima to the reaktoro/core directory
add_custom_command(TARGET PyOptima POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:PyOptima>
    ${CMAKE_BINARY_DIR}/OptimaPython/reaktoro/core/$<TARGET_FILE_NAME:PyOptima>
    COMMENT "Copying the Python extension module reaktoro to the reaktoro/core directory.")
