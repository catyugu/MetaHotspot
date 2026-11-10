add_library(mhs_compiler STATIC
    discrete_mesh.cpp
    conformal_mesh.cpp
    volume_rasterizer.cpp
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    model_compiler.cpp
    model_physics.cpp
    model_functions.cpp
)
target_link_libraries(mhs_compiler
    PUBLIC mhs_model
    PRIVATE mhs_linear mhs_logging
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)

