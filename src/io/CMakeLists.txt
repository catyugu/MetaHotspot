add_library(io_lib io.cpp)
target_link_libraries(io_lib PUBLIC tinyxml2 common_lib)
target_include_directories(io_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
