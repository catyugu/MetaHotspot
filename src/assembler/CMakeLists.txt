add_library(assembler_lib STATIC assembler.cpp)
target_link_libraries(assembler_lib PRIVATE TBB::tbb common_lib)
target_link_libraries(assembler_lib PUBLIC Eigen3::Eigen)
target_include_directories(assembler_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
