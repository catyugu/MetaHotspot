include(GoogleTest)

# ==============================================================================
# 单一测试可执行文件
# ==============================================================================

file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(mhs_tests
    ${test_sources}
)

target_link_libraries(mhs_tests PRIVATE
    mhs_lib
    gtest_main
)

target_include_directories(mhs_tests PRIVATE ${CMAKE_BINARY_DIR})

mhs_copy_runtime_dlls(mhs_tests)
gtest_discover_tests(mhs_tests)
