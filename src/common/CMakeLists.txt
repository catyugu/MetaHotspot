add_library(common_lib STATIC logger.cpp sample_point.cpp)
target_link_libraries(common_lib
    PUBLIC spdlog
    PRIVATE expr_lib Eigen3::Eigen
)
target_include_directories(common_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
