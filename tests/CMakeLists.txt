include(GoogleTest)

# One test executable; new tests/*.cpp files are picked up on reconfiguration.
file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(mhs_tests ${test_sources})
# Test data lives in the source tree, so the tests address it through this.
target_compile_definitions(mhs_tests PRIVATE MHS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# White-box tests reach component-internal headers that are not installed API.
target_include_directories(mhs_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(mhs_tests PRIVATE
    mhs_metahotspot
    GTest::gtest_main
)

if(MHS_ENABLE_PARDISO)
    target_compile_definitions(mhs_tests PRIVATE MHS_ENABLE_PARDISO)
endif()

mhs_stage_runtime(mhs_tests)

# Discovery runs once per build (the default POST_BUILD mode). PRE_TEST re-runs it
# on every ctest invocation and also wraps each test in a `cmake -P` process,
# which costs ~70 ms — more than any test here. The default 5 s is measured from
# process start, so a cold Debug executable on a loaded CI runner runs out.
gtest_discover_tests(mhs_tests
    TEST_PREFIX "mhs."
    DISCOVERY_TIMEOUT 60
)
