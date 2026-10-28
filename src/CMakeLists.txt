add_subdirectory(numerics)
add_subdirectory(logging)
add_subdirectory(compiler)
add_subdirectory(solver)
add_subdirectory(io)
if(MHS_BUILD_C_API)
    add_subdirectory(api)
endif()

# Every component exposes the public headers under include/metahotspot; a
# component's own directory is only for its implementation headers.
foreach(_component expression linear compiler solver io logging)
    add_library(mhs::${_component} ALIAS mhs_${_component})
    set_target_properties(mhs_${_component} PROPERTIES EXPORT_NAME ${_component})
    target_compile_features(mhs_${_component} PUBLIC cxx_std_20)
    target_include_directories(mhs_${_component} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/metahotspot>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/metahotspot>
    )
endforeach()
