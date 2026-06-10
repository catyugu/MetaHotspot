add_library(preprocessor_lib STATIC
    preprocessor.cpp
    layer_processor.cpp
    face_key_processor.cpp
    function_helpers.cpp
)

target_link_libraries(preprocessor_lib PRIVATE expr_lib common_lib io_lib)

target_include_directories(preprocessor_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
