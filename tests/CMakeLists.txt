include(GoogleTest)

if(MHS_BUILD_UNIT_TESTS)
    add_subdirectory(unit)
endif()
if(MHS_BUILD_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()
