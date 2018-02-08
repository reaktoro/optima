# Collect the header and source files of PyOptima
file(GLOB_RECURSE HEADER_FILES *.hpp)
file(GLOB_RECURSE SOURCE_FILES *.cpp)

# Combine PyOptima and Optima source files to compile the python module
pybind11_add_module(optima ${SOURCE_FILES} $<TARGET_OBJECTS:OptimaObject>)
#pybind11_add_module(optima ${SOURCE_FILES})
#pybind11_add_module(optima PyOptima.cpp $<TARGET_OBJECTS:OptimaObject>)

