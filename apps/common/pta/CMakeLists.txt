add_library(mhs_pta_model_io STATIC
    definition.cpp
    geometry_expression.cpp
    xml_reader.cpp
    face_region_parser.cpp
)
target_link_libraries(mhs_pta_model_io
    PUBLIC mhs::metahotspot
    PRIVATE mhs_tinyxml2_headers mhs_options
)
target_include_directories(mhs_pta_model_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
