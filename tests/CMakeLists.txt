set(OPTIMA_LIBRARY_DIR ${CMAKE_BINARY_DIR}/Optima)
set(OPTIMA_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/package/build/lib)

file(TO_NATIVE_PATH ${OPTIMA_LIBRARY_DIR} OPTIMA_LIBRARY_DIR)
file(TO_NATIVE_PATH ${OPTIMA_PYTHON_PACKAGE_DIR} OPTIMA_PYTHON_PACKAGE_DIR)

if(WIN32)
    add_custom_target(tests
        COMMAND set "PATH=${OPTIMA_LIBRARY_DIR}\\$<CONFIG>;$ENV{PATH}"
        COMMAND set "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR};$ENV{PYTHONPATH}"
        COMMAND ${CMAKE_COMMAND} -E echo "Ensure python package optima exists in ${OPTIMA_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E echo "Ensure Optima C++ library exists in ${OPTIMA_LIBRARY_DIR}\\$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E echo "Starting parallel execution of pytest..."
        COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else() # UNIX
    add_custom_target(tests
        COMMAND ${CMAKE_COMMAND} -E echo "Ensure python package optima exists in ${OPTIMA_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E echo "Ensure Optima C++ library exists in ${OPTIMA_LIBRARY_DIR}"
        COMMAND export "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR}:$ENV{PYTHONPATH}"
        COMMAND ${CMAKE_COMMAND} -E echo "Starting parallel execution of pytest..."
        COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
