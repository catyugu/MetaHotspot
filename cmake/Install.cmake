# Install rules and CMake package export. Included from the top-level
# CMakeLists.txt when MHS_INSTALL is on.

include(CMakePackageConfigHelpers)

set(_mhs_targets mhs_metahotspot mhs_core mhs_model mhs_expression mhs_linear
    mhs_compiler mhs_solver mhs_logging)
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

# muparser's upstream install rule hardcodes include/muParser*.h. Move those
# private headers after that rule has run; the exported include directory was
# adjusted to match in Dependencies.cmake. Honor DESTDIR for CPack staging.
if(TARGET muparser)
    get_target_property(_mhs_muparser_imported muparser IMPORTED)
    if(NOT _mhs_muparser_imported)
        set(_mhs_relocate_muparser [=[
            set(_mhs_prefix "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
            file(GLOB _mhs_muparser_headers "${_mhs_prefix}/include/muParser*.h")
            if(_mhs_muparser_headers)
                set(_mhs_destination "${_mhs_prefix}/@CMAKE_INSTALL_INCLUDEDIR@/third_party/muparser")
                file(MAKE_DIRECTORY "${_mhs_destination}")
                foreach(_mhs_header IN LISTS _mhs_muparser_headers)
                    get_filename_component(_mhs_name "${_mhs_header}" NAME)
                    file(RENAME "${_mhs_header}" "${_mhs_destination}/${_mhs_name}")
                endforeach()
            endif()
        ]=])
        string(CONFIGURE "${_mhs_relocate_muparser}" _mhs_relocate_muparser @ONLY)
        install(CODE "${_mhs_relocate_muparser}")
        unset(_mhs_relocate_muparser)
    endif()
endif()

if(MHS_BUILD_APPS)
    install(TARGETS metahotspot RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Runtime DLLs of the installed binaries; the MSVC C runtime is not among them.
mhs_install_windows_runtime(TARGETS metahotspot mhs_c_api
    DESTINATION ${CMAKE_INSTALL_BINDIR})

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
