add_library(mhs_compiler STATIC
    cartesian_mesh.cpp
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    model_compiler.cpp
    model_functions.cpp
)
target_link_libraries(mhs_compiler
    PUBLIC mhs_expression
    PRIVATE mhs_linear
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
