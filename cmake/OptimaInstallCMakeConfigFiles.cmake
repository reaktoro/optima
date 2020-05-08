# The path where cmake config files are installed
set(OPTIMA_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Optima)

install(EXPORT OptimaTargets
    FILE OptimaTargets.cmake
    NAMESPACE Optima::
    DESTINATION ${OPTIMA_INSTALL_CONFIGDIR}
    COMPONENT cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OptimaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OptimaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OptimaConfig.cmake
    INSTALL_DESTINATION ${OPTIMA_INSTALL_CONFIGDIR}
    PATH_VARS OPTIMA_INSTALL_CONFIGDIR)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OptimaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OptimaConfigVersion.cmake
    DESTINATION ${OPTIMA_INSTALL_CONFIGDIR})
