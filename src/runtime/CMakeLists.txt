add_library(mhs_runtime INTERFACE)

target_include_directories(mhs_runtime INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mhs_runtime INTERFACE mhs_expression)
