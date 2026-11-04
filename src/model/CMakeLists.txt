add_library(mhs_model STATIC discrete_mesh.cpp)
target_link_libraries(mhs_model PUBLIC mhs_core mhs_expression
    PRIVATE $<BUILD_INTERFACE:mhs_options>)
