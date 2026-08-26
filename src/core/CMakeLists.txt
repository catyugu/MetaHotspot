add_library(mhs_core INTERFACE)

target_include_directories(mhs_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mhs_core INTERFACE mhs_expression)
