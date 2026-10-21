add_subdirectory(numerics)
add_subdirectory(logging)
add_subdirectory(core)
add_subdirectory(compiler)
add_subdirectory(solver)
add_subdirectory(io)
if(MHS_BUILD_C_API)
    add_subdirectory(api)
endif()

foreach(_component core expression linear compiler solver io logging)
    add_library(mhs::${_component} ALIAS mhs_${_component})
    set_target_properties(mhs_${_component} PROPERTIES EXPORT_NAME ${_component})
    target_include_directories(mhs_${_component} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/metahotspot>
    )
endforeach()
