add_library(mhs_compiler STATIC
    conformal_mesh.cpp
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    fom_compiler.cpp
    model_functions.cpp
    project_compiler.cpp
)
target_link_libraries(mhs_compiler
    PUBLIC mhs_model
    PRIVATE mhs_linear
    PRIVATE mhs_logging
    PRIVATE $<BUILD_INTERFACE:Eigen3::Eigen>
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
