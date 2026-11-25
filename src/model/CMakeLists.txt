add_library(mhs_model STATIC erom.cpp project.cpp $<TARGET_OBJECTS:mhs_tinyxml2>)
target_link_libraries(mhs_model
    PUBLIC mhs_core mhs_expression
    PRIVATE $<BUILD_INTERFACE:mhs_tinyxml2_headers> $<BUILD_INTERFACE:Eigen3::Eigen> $<BUILD_INTERFACE:mhs_options>
)
