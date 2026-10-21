add_library(mhs_c_api SHARED
    metahotspot.cpp
)
target_link_libraries(mhs_c_api PRIVATE
    mhs_compiler
    mhs_solver
    mhs_io
)
target_include_directories(mhs_c_api
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/metahotspot>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/metahotspot>
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(mhs_c_api PRIVATE
    MHS_API_EXPORT
)
target_link_libraries(mhs_c_api PRIVATE $<BUILD_INTERFACE:mhs_options>)

target_compile_definitions(mhs_c_api PUBLIC MHS_BUILD_SHARED)
add_library(mhs::c_api ALIAS mhs_c_api)
# Every symbol is exported by default on ELF/Mach-O; hiding them makes the
# MHS_API attribute on the public header the only thing that escapes.
set_target_properties(mhs_c_api PROPERTIES
    EXPORT_NAME c_api
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${MHS_VERSION}
    SOVERSION ${MHS_ABI_VERSION}
)
if(APPLE)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "$ORIGIN")
endif()
mhs_stage_runtime(mhs_c_api)
