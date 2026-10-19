add_library(mhs_core INTERFACE)

target_include_directories(mhs_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(mhs_core INTERFACE mhs_expression)
