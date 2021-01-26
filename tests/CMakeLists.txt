set(OPTIMA_LIBRARY_DIR ${CMAKE_BINARY_DIR}/Optima)  # the directory where the libraries are compiled
set(OPTIMA_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/package/build/lib)  # the directory where the python package is built

file(TO_NATIVE_PATH ${OPTIMA_LIBRARY_DIR} OPTIMA_LIBRARY_DIR)  # correct slashes if in Windows
file(TO_NATIVE_PATH ${OPTIMA_PYTHON_PACKAGE_DIR} OPTIMA_PYTHON_PACKAGE_DIR)  # correct slashes if in Windows

if(WIN32)
    add_custom_target(tests
        COMMENT "Ensure python package optima exists in ${OPTIMA_PYTHON_PACKAGE_DIR}"
        COMMENT "Ensure Optima C++ library exists in ${OPTIMA_LIBRARY_DIR}\\$<CONFIG>"
        COMMENT "Starting parallel execution of pytest..."
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=${OPTIMA_LIBRARY_DIR}\\$<CONFIG>;$ENV{PATH}"
            "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR}"
                pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else() # UNIX
    add_custom_target(tests
        COMMENT "Ensure python package optima exists in ${OPTIMA_PYTHON_PACKAGE_DIR}"
        COMMENT "Ensure Optima C++ library exists in ${OPTIMA_LIBRARY_DIR}"
        COMMENT "Starting parallel execution of pytest..."
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR}"
                pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
