# Install rules and CMake package export. Included from the top-level
# CMakeLists.txt when MHS_INSTALL is on.

include(CMakePackageConfigHelpers)

set(_mhs_targets mhs_metahotspot mhs_expression mhs_linear
    mhs_compiler mhs_solver mhs_io mhs_logging)
if(MHS_BUILD_C_API)
    list(APPEND _mhs_targets mhs_c_api)
endif()

install(TARGETS ${_mhs_targets} EXPORT MetaHotspotTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# include/metahotspot holds the public API and src/ the implementation, so the
# whole public tree is installed as it stands.
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/metahotspot"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(MHS_BUILD_APPS)
    install(TARGETS metahotspot RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Licence and readme ship with the installed package.
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/README.md"
    DESTINATION ${CMAKE_INSTALL_DOCDIR})

install(EXPORT MetaHotspotTargets NAMESPACE mhs::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)

configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/MetaHotspotConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/MetaHotspotConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)
# Same compatibility boundary as the shared library SONAME.
write_basic_package_version_file("${PROJECT_BINARY_DIR}/MetaHotspotConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${PROJECT_BINARY_DIR}/MetaHotspotConfig.cmake"
    "${PROJECT_BINARY_DIR}/MetaHotspotConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetaHotspot)

# Package the install tree, including CPM-built dependency install rules.
if(PROJECT_IS_TOP_LEVEL)
    set(CPACK_PACKAGE_NAME MetaHotspot)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_GENERATOR ZIP)
    include(CPack)
endif()
