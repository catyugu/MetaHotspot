set(mhs_io_sources
    xml_model_reader.cpp
    xml_writer.cpp
    vtu_writer.cpp
    face_region_parser.cpp
)

add_library(mhs_io STATIC ${mhs_io_sources})
target_include_directories(mhs_io PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_io
    PUBLIC mhs_core
    PRIVATE tinyxml2::tinyxml2
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)

mhs_install_headers(io model_io.hpp result_io.hpp)
