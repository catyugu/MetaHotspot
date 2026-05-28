add_executable(metahotspot ../src/main.cpp)
target_link_libraries(metahotspot PRIVATE mhs_lib)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)