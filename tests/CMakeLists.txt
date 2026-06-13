include(GoogleTest)

# ==============================================================================
# 创建一个统一的测试基础 INTERFACE 库
# ==============================================================================
# 不编译任何代码，专门用来集中管理所有测试文件共同依赖的库、宏定义和包含路径
add_library(mhs_test_base INTERFACE)

target_link_libraries(mhs_test_base INTERFACE
    mhs_lib
    gtest_main
)

target_include_directories(mhs_test_base INTERFACE ${CMAKE_BINARY_DIR})

function(mhs_add_test_executable TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE mhs_test_base)
    mhs_copy_runtime_dlls(${TEST_NAME})
    gtest_discover_tests(${TEST_NAME})
endfunction()

# ==============================================================================
# 声明测试目标
# ==============================================================================
mhs_add_test_executable(test_expr test_expr.cpp)
mhs_add_test_executable(test_preprocessor test_preprocessor.cpp)
mhs_add_test_executable(test_postprocessor test_postprocessor.cpp)
mhs_add_test_executable(test_assembler test_assembler.cpp)
mhs_add_test_executable(test_scheduler test_scheduler.cpp)
mhs_add_test_executable(test_logger test_logger.cpp)
mhs_add_test_executable(test_function_helpers test_function_helpers.cpp)
mhs_add_test_executable(test_io test_io.cpp)
