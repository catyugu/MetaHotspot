add_library(mhs_compiler STATIC
    fluid_preprocessor.cpp
    geometry_compiler.cpp
    model_compiler.cpp
    model_functions.cpp
)
target_include_directories(mhs_compiler PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_compiler
    PUBLIC mhs_common
    PRIVATE mhs_linear
    PRIVATE mhs_options
)
