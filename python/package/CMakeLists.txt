# The name of the compiled python module file as defined by pybind11 used in setup.py.in
set(OPTIMA_PYTHON_MODULE_FILENAME optima4py${PYTHON_MODULE_EXTENSION})

# The path in the binary dir where the python package is assembled
set(OPTIMA_PYTHON_PACKAGE_PATH ${CMAKE_CURRENT_BINARY_DIR}/optima)

# Configure the setup.py file
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY ${OPTIMA_PYTHON_PACKAGE_PATH}/setup.py) # ensure setup.py is inside the package dir
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# Create a custom target to build the python package during build stage
add_custom_target(optima-setuptools ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/optima ${OPTIMA_PYTHON_PACKAGE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:optima4py> ${OPTIMA_PYTHON_PACKAGE_PATH}
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} --quiet install --force --prefix=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${OPTIMA_PYTHON_PACKAGE_PATH})

# Set dependencies of the python package target
add_dependencies(optima-setuptools optima4py)

# Set the path where the python package is installed to CMAKE_INSTALL_PREFIX if not given
if(NOT DEFINED OPTIMA_PYTHON_INSTALL_PREFIX)
    set(OPTIMA_PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# If the path is already in Windows format (with backslashes), it can't be
# added directly to the string below. CMake complains about "Invalid escape sequence".
file(TO_CMAKE_PATH "${OPTIMA_PYTHON_INSTALL_PREFIX}" OPTIMA_PYTHON_INSTALL_PREFIX)

# Install the python package using setuptools
install(CODE
"
    file(TO_NATIVE_PATH \"${OPTIMA_PYTHON_INSTALL_PREFIX}\" OPTIMA_PYTHON_INSTALL_PREFIX_NATIVE)

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=\${OPTIMA_PYTHON_INSTALL_PREFIX_NATIVE}
        WORKING_DIRECTORY ${OPTIMA_PYTHON_PACKAGE_PATH})
")
