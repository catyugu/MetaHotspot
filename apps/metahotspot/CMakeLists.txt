add_library(metahotspot_frontend STATIC
    output/vtu_writer.cpp
    postprocess.cpp
)
target_link_libraries(metahotspot_frontend
    PUBLIC mhs::metahotspot mhs_pta_io
    PRIVATE tinyxml2::tinyxml2 mhs_options
)
target_include_directories(metahotspot_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(metahotspot main.cpp cli.cpp)
target_link_libraries(metahotspot PRIVATE metahotspot_frontend mhs_options)
mhs_stage_runtime(metahotspot)
if(APPLE)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(MHS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
