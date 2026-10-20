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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(mhs_c_api PRIVATE
    MHS_API_EXPORT
)
target_link_libraries(mhs_c_api PRIVATE $<BUILD_INTERFACE:mhs_options>)
mhs_copy_runtime_dlls(mhs_c_api)

target_compile_definitions(mhs_c_api PUBLIC MHS_BUILD_SHARED)
add_library(mhs::c_api ALIAS mhs_c_api)
set_target_properties(mhs_c_api PROPERTIES EXPORT_NAME c_api VERSION ${MHS_VERSION} SOVERSION 1)
if(APPLE)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "$ORIGIN")
endif()
