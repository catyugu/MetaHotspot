add_library(assembler_lib STATIC assembler.cpp)
target_link_libraries(assembler_lib PRIVATE Eigen3::Eigen TBB::tbb)
target_include_directories(assembler_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
