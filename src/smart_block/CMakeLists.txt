mhs_add_library(smart_block_lib
    SOURCES smart_block_reader.cpp
    PUBLIC_LINKS Eigen3::Eigen
)

target_include_directories(smart_block_lib PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
