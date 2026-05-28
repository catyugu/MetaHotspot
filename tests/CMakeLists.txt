# 移除这里的 enable_testing()

add_executable(test_model test_model.cpp)
target_link_libraries(test_model mhs_lib gtest_main)

add_executable(test_expr test_expr.cpp)
target_link_libraries(test_expr mhs_lib gtest_main)

add_executable(test_preprocessor test_preprocessor.cpp)
target_link_libraries(test_preprocessor mhs_lib gtest_main)

add_executable(test_assembler test_assembler.cpp)
target_link_libraries(test_assembler mhs_lib gtest_main)

add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler mhs_lib gtest_main)

include(GoogleTest)

gtest_discover_tests(test_model)
gtest_discover_tests(test_expr)
gtest_discover_tests(test_preprocessor)
gtest_discover_tests(test_assembler)
gtest_discover_tests(test_scheduler)
