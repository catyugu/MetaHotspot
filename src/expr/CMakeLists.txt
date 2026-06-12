add_library(expr_lib STATIC expr.cpp)

target_link_libraries(expr_lib PRIVATE TBB::tbb muparser)

target_include_directories(expr_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
