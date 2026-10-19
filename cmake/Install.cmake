include(CMakePackageConfigHelpers)

get_target_property(_muparser_links muparser INTERFACE_LINK_LIBRARIES)
set(MHS_NEEDS_OPENMP OFF)
if("${_muparser_links}" MATCHES "OpenMP::OpenMP_CXX")
    set(MHS_NEEDS_OPENMP ON)
endif()

set(_mhs_targets mhs_metahotspot mhs_core mhs_expression mhs_amg mhs_linear
    mhs_compiler mhs_solver mhs_io mhs_logging)
if(MHS_BUILD_C_API)
    list(APPEND _mhs_targets mhs_c_api)
endif()
install(TARGETS ${_mhs_targets} EXPORT MetaHotspotTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metahotspot
    FILES_MATCHING PATTERN "*.hpp")
if(MHS_BUILD_TOOLS)
    install(TARGETS metahotspot RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES "${PROJECT_SOURCE_DIR}/README.md" DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(EXPORT MetaHotspotTargets NAMESPACE mhs::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/MetaHotspotConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/MetaHotspotConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/MetaHotspotConfigVersion.cmake"
    VERSION ${MHS_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${PROJECT_BINARY_DIR}/MetaHotspotConfig.cmake"
    "${PROJECT_BINARY_DIR}/MetaHotspotConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)

# Package the install tree, including CPM-built dependency install rules.
if(PROJECT_IS_TOP_LEVEL)
    set(CPACK_PACKAGE_NAME MetaHotspot)
    set(CPACK_PACKAGE_VERSION ${MHS_VERSION})
    set(CPACK_GENERATOR ZIP)
    include(CPack)
endif()
