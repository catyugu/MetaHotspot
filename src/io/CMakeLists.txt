add_library(mhs_io STATIC
    xml_model_reader.cpp
    xml_fluid_reader.cpp
    xml_writer.cpp
    vtu_writer.cpp
    face_region_parser.cpp
)
target_include_directories(mhs_io PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_io
    PUBLIC mhs_core
    PRIVATE tinyxml2
    PRIVATE mhs_options
)

target_include_directories(mhs_io SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/tinyxml2-src
)
