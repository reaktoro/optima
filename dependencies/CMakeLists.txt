# Include the ExternalProject module
include(ExternalProject)

# Set the default build type to Release for the external projects
if(NOT EXTERNAL_BUILD_TYPE)
    # The build type selection for the external projects
    set(EXTERNAL_BUILD_TYPE Release CACHE STRING "Choose the build type for the external projects." FORCE)
    
    # The build type options for the external projects
    set_property(CACHE EXTERNAL_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

# Output a message detailing the build type used for the external projects
message(STATUS "Using build type `${EXTERNAL_BUILD_TYPE}` for the external projects.")

# Set the path where the external projects should be installed for the compitation of this project
if(NOT EXTERNAL_INSTALL_PREFIX)
    set(EXTERNAL_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/external)
endif()

# Set the intermediate install directory for the external projects
set(EXTERNAL_INTERMEDIATE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

# Set the common cmake arguments to all external projects
set(EXTERNAL_CMAKE_ARGS
    -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INTERMEDIATE_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${EXTERNAL_BUILD_TYPE}
    -DCMAKE_C_FLAGS=${CFLAGS}
    -DCMAKE_CXX_FLAGS=${CXXFLAGS}
    -DCMAKE_INSTALL_INCLUDEDIR=include
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_INSTALL_BINDIR=bin
     # Disables the cmake warning message `Manually-specified
     # variables were not used by the project` because of
     # unused arguments above
    --no-warn-unused-cli
    )

# Build the Eigenx library and do an intermediate install to EXTERNAL_INTERMEDIATE_INSTALL_DIR
ExternalProject_Add(EIGENX
    PREFIX build
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Eigenx
    CMAKE_ARGS ${EXTERNAL_CMAKE_ARGS})

# Copy the installed include directories from the EIGEN target and install under external dir
add_custom_command(TARGET EIGENX POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${EXTERNAL_INTERMEDIATE_INSTALL_DIR}/include
        ${EXTERNAL_INSTALL_PREFIX}/include/Optima/external
        )