add_library(time_scheme_lib STATIC
    time_scheme.cpp
)

target_link_libraries(time_scheme_lib PUBLIC common_lib Eigen3::Eigen)
target_include_directories(time_scheme_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
