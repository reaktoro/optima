file(GLOB CPPFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

include_directories(${PROJECT_SOURCE_DIR})

foreach(CPPFILE ${CPPFILES})
    get_filename_component(CPPNAME ${CPPFILE} NAME_WE)
    add_executable(${CPPNAME} $<TARGET_OBJECTS:OptimaObject> ${CPPFILE})
endforeach()
