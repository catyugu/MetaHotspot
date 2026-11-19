add_library(mhs_compiler STATIC
    discrete_mesh.cpp
    conformal_mesh.cpp
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    model_compiler.cpp
    model_functions.cpp
    project.cpp
    erom.cpp
)
target_link_libraries(mhs_compiler
    PUBLIC mhs_model
    PRIVATE mhs_linear
    PRIVATE mhs_logging
    PRIVATE $<BUILD_INTERFACE:Eigen3::Eigen>
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
