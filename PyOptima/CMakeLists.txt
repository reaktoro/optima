# Collect the source files of PyOptima
file(GLOB_RECURSE SOURCE_FILES Py*.cpp)

# Combine PyOptima and Optima source files to compile the python module
pybind11_add_module(optima ${SOURCE_FILES} $<TARGET_OBJECTS:OptimaObject>)
