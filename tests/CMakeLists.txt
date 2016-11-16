# Include the current directory so that cute test framework can be used
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB CPPFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(CPPFILE ${CPPFILES})
    get_filename_component(CPPNAME ${CPPFILE} NAME_WE)
    add_executable(${CPPNAME} ${CPPFILE})
    target_link_libraries(${CPPNAME} OptimaShared)
endforeach()
