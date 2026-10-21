set(mhs_compiler_sources
    cartesian_mesh.cpp
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    model_compiler.cpp
    model_functions.cpp
)

add_library(mhs_compiler STATIC ${mhs_compiler_sources})
target_include_directories(mhs_compiler PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_compiler
    PUBLIC mhs_core
    PRIVATE mhs_linear
    PRIVATE $<BUILD_INTERFACE:mhs_options>
)
mhs_install_headers(compiler model_compiler.hpp)
