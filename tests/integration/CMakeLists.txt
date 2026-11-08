file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(mhs_integration_tests ${test_sources})
target_compile_definitions(mhs_integration_tests PRIVATE MHS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_include_directories(mhs_integration_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_integration_tests PRIVATE mhs_metahotspot GTest::gtest_main)
if(MHS_ENABLE_PARDISO)
    target_compile_definitions(mhs_integration_tests PRIVATE MHS_ENABLE_PARDISO)
endif()
mhs_stage_runtime(mhs_integration_tests)
gtest_discover_tests(mhs_integration_tests
    TEST_PREFIX "mhs.integration."
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "integration"
)

if(MHS_BUILD_C_API)
    enable_language(C)
    add_executable(mhs_c_layout_test test_c_layout.c)
    # Keep the client compiled as C; the C++ driver supplies the shared library's
    # C++ UBSan runtime (including vptr checks) in sanitizer builds.
    set_target_properties(mhs_c_layout_test PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON LINKER_LANGUAGE CXX)
    target_link_libraries(mhs_c_layout_test PRIVATE mhs_c_api)
    mhs_stage_runtime(mhs_c_layout_test)
    add_test(NAME mhs.integration.c_layout COMMAND mhs_c_layout_test)
    set_tests_properties(mhs.integration.c_layout PROPERTIES LABELS integration)
endif()
