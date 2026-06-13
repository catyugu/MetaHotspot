add_library(time_scheme_lib STATIC
    time_scheme.cpp
    bdf1_scheme.cpp
    bdf2_scheme.cpp
    adaptive_bdf_scheme.cpp
)
target_link_libraries(time_scheme_lib PUBLIC assembler_lib)
target_include_directories(time_scheme_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
