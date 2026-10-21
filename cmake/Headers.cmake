# Public header layout.
#
# A component names the headers that belong to its API; they are installed under
# <prefix>/include/metahotspot/<subdir>/ and reachable through the exported
# targets. Implementation headers are never named, so a header is installed
# because a component said so, never because it happens to live under src/.
function(mhs_install_headers subdir)
    if(MHS_INSTALL)
        install(FILES ${ARGN}
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/metahotspot/${subdir}")
    endif()
endfunction()
