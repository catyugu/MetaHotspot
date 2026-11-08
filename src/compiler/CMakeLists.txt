add_library(mhs_compiler STATIC
    discrete_mesh.cpp
    cartesian_mesh.cpp
    geometry_compiler.cpp
    connections.cpp
    model_compiler.cpp
    model_functions.cpp
)
target_link_libraries(mhs_compiler
    PUBLIC mhs_model
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
