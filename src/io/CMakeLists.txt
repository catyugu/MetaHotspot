mhs_add_library(io_lib
    SOURCES xml_reader.cpp xml_writer.cpp vtu_writer.cpp
    PRIVATE_LINKS logger_lib tinyxml2 Eigen3::Eigen
)

# tinyxml2 is fetched via CPM and its headers are exposed at
# ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src rather than via a clean
# INTERFACE include directory. io.cpp includes <tinyxml2.h>
target_include_directories(io_lib PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
