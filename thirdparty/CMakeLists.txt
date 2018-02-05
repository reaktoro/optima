add_subdirectory(catch EXCLUDE_FROM_ALL)

# Add target "demos" for manual building of demos, as `make demos`, if BUILD_DEMOS is OFF
add_custom_target(catch
    COMMAND ${CMAKE_MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/catch")