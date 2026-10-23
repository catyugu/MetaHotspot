add_library(mhs_io STATIC
    xml_model_reader.cpp
    xml_writer.cpp
    vtu_writer.cpp
    face_region_parser.cpp
)
# The tinyxml2 wrappers are implementation details and stay in this directory.
target_include_directories(mhs_io PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_io
    PUBLIC mhs_expression
    PRIVATE tinyxml2::tinyxml2
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
