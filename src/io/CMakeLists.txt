add_library(io_lib io.cpp)
target_link_libraries(io_lib
    PRIVATE common_lib tinyxml2
)
target_include_directories(io_lib PUBLIC
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
