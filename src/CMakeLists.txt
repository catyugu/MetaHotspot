add_library(mhs_lib INTERFACE)
target_include_directories(mhs_lib INTERFACE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
)
