mhs_add_library(preprocessor_lib
    SOURCES preprocessor.cpp layer_processor.cpp boundary_processor.cpp function_helpers.cpp
    PUBLIC_LINKS mhs_model_lib
    PRIVATE_LINKS expr_lib fluid_lib utils_lib logger_lib
)
