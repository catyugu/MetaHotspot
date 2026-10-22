include(GoogleTest)

# One test executable; new tests/*.cpp files are picked up on reconfiguration.
file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(mhs_tests ${test_sources})
# Test data lives in the source tree, so the tests address it through this.
target_compile_definitions(mhs_tests PRIVATE MHS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

target_link_libraries(mhs_tests PRIVATE
    mhs_metahotspot
    GTest::gtest_main
)

mhs_stage_runtime(mhs_tests)

gtest_discover_tests(mhs_tests
    TEST_PREFIX "mhs."
    DISCOVERY_MODE PRE_TEST
)
