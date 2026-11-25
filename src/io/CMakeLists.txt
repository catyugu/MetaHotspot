add_library(mhs_io STATIC erom.cpp $<TARGET_OBJECTS:mhs_tinyxml2>)
target_link_libraries(mhs_io
    PUBLIC mhs_model
    PRIVATE $<BUILD_INTERFACE:mhs_tinyxml2_headers> $<BUILD_INTERFACE:mhs_options>
)
