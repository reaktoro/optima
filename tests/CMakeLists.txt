set(OPTIMA_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/package/build/lib)  # the directory where the python package is built

file(TO_NATIVE_PATH ${OPTIMA_PYTHON_PACKAGE_DIR} OPTIMA_PYTHON_PACKAGE_DIR)  # correct slashes if in Windows

if(WIN32)
    add_custom_target(tests
        DEPENDS optima-setuptools
        COMMENT "Starting parallel execution of pytest..."
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=$<TARGET_FILE_DIR:Optima>;$ENV{PATH}"
            "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR}"
                pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else() # UNIX
    add_custom_target(tests
        DEPENDS optima-setuptools
        COMMENT "Starting parallel execution of pytest..."
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${OPTIMA_PYTHON_PACKAGE_DIR}"
                pytest ${CMAKE_CURRENT_SOURCE_DIR} -n auto -x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
