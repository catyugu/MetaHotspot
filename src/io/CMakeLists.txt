add_library(mhs_io STATIC erom.cpp)
target_link_libraries(mhs_io
    PUBLIC mhs_model
    PRIVATE tinyxml2::tinyxml2 $<BUILD_INTERFACE:mhs_options>
)
