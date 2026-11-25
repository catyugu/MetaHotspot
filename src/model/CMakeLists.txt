add_library(mhs_model STATIC erom.cpp project.cpp)
target_link_libraries(mhs_model
    PUBLIC mhs_core mhs_expression
    PRIVATE tinyxml2::tinyxml2 $<BUILD_INTERFACE:Eigen3::Eigen> $<BUILD_INTERFACE:mhs_options>
)
