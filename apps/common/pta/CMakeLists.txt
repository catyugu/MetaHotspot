add_library(mhs_pta_io STATIC
    definition.cpp
    io.cpp
)
target_link_libraries(mhs_pta_io
    PUBLIC mhs::metahotspot
    PRIVATE mhs_tinyxml2_headers mhs_options
)
target_include_directories(mhs_pta_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
