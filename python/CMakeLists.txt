# Collect all source files from the current directory for the Python module 
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile Optima cxx files into a Python module
pybind11_add_module(optima ${CXX_FILES} $<TARGET_OBJECTS:OptimaObject>)
#pybind11_add_module(optima ${CXX_FILES})
#target_link_libraries(optima LINK_PRIVATE OptimaStatic)
