mhs_add_library(preprocessor_lib
    SOURCES preprocessor.cpp model_builder.cpp layer_processor.cpp face_key_processor.cpp function_helpers.cpp
            fluid_preprocessor.cpp fluid_flow_solver.cpp
    PRIVATE_LINKS expr_lib utils_lib io_lib linear_solver_lib logger_lib
)

target_link_libraries(preprocessor_lib PRIVATE Eigen3::Eigen)
