mhs_add_library(preprocessor_lib
    SOURCES preprocessor.cpp layer_processor.cpp face_key_processor.cpp function_helpers.cpp fluid_preprocessor.cpp
    PRIVATE_LINKS expr_lib common_lib io_lib linear_solver_lib
)

target_link_libraries(preprocessor_lib PRIVATE Eigen3::Eigen)
