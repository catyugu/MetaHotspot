mhs_add_library(mhs_io
    SOURCES xml_reader.cpp xml_writer.cpp vtu_writer.cpp face_region_parser.cpp
    PUBLIC_LINKS mhs_model
    PRIVATE_LINKS mhs_logging tinyxml2
)

target_include_directories(mhs_io SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
