include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/thirdparty)

file(GLOB_RECURSE CPPFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} Test*.cpp)

foreach(CPPFILE ${CPPFILES})
    get_filename_component(CPPNAME ${CPPFILE} NAME_WE)
    add_executable(${CPPNAME} ${CPPFILE} $<TARGET_OBJECTS:OptimaObject> $<TARGET_OBJECTS:CatchObject>)
    add_test(NAME ${CPPNAME} COMMAND ${CPPNAME})
endforeach()
