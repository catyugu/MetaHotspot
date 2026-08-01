add_library(mhs_common INTERFACE)

target_include_directories(mhs_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mhs_common INTERFACE mhs_expression)
