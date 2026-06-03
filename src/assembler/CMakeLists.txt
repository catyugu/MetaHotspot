add_library(assembler_lib STATIC assembler.cpp)
target_link_libraries(assembler_lib PRIVATE Eigen3::Eigen TBB::tbb common_lib)
target_include_directories(assembler_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)

add_custom_command(TARGET assembler_lib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TBB::tbb>
    $<TARGET_FILE_DIR:assembler_lib>
)
