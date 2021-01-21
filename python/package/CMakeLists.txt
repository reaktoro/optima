# The name of the compiled python module file as defined by pybind11 used in setup.py.in
set(OPTIMA_PYTHON_MODULE_FILENAME optima4py${PYTHON_MODULE_EXTENSION})

# The path in the binary dir where the python package is assembled before it is built
set(OPTIMA_PYTHON_PACKAGE_PATH ${CMAKE_CURRENT_BINARY_DIR}/optima)

# Configure the setup.py file
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# This is needed in Windows so that the next step produces the python package in CMAKE_BINARY_DIR\Lib\site-packages\optima
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} CMAKE_BINARY_DIR_NATIVE)

# Create a custom target to build the python package during build stage
add_custom_target(optima-setuptools ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/optima ${CMAKE_CURRENT_BINARY_DIR}/optima
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:optima4py> ${CMAKE_CURRENT_BINARY_DIR}/optima
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} --quiet install --force --no-compile --prefix=installed/$<CONFIG>
    COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_BINARY_DIR}/optima/$<TARGET_FILE_NAME:optima4py>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Set dependencies of the python package target
add_dependencies(optima-setuptools optima4py)

# Ensure the path where the python package is installed is not empty
if(NOT DEFINED OPTIMA_PYTHON_INSTALL_PREFIX)
    set(OPTIMA_PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# Create an install target for the optima python package
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/installed/$<CONFIG>/
    DESTINATION ${OPTIMA_PYTHON_INSTALL_PREFIX})
