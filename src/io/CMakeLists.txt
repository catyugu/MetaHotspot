mhs_add_library(io_lib
    SOURCES io.cpp
    PRIVATE_LINKS common_lib tinyxml2 Eigen3::Eigen
)

# tinyxml2 is fetched via CPM and its headers are exposed at
# ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src rather than via a clean
# INTERFACE include directory. io.cpp includes <tinyxml2.h>
target_include_directories(io_lib PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
