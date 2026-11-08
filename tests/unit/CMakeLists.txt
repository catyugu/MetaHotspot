file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(mhs_unit_tests ${test_sources})
target_compile_definitions(mhs_unit_tests PRIVATE MHS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_include_directories(mhs_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mhs_unit_tests PRIVATE mhs_metahotspot GTest::gtest_main)
if(MHS_ENABLE_PARDISO)
    target_compile_definitions(mhs_unit_tests PRIVATE MHS_ENABLE_PARDISO)
endif()
mhs_stage_runtime(mhs_unit_tests)
gtest_discover_tests(mhs_unit_tests
    TEST_PREFIX "mhs.unit."
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "unit"
)
