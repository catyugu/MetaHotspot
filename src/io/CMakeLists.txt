mhs_add_library(io_lib
    SOURCES xml_reader.cpp xml_writer.cpp vtu_writer.cpp
    PRIVATE_LINKS logger_lib tinyxml2
)

target_include_directories(io_lib SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
