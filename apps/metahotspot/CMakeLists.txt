add_library(mhs_app_support STATIC
    postprocessor.cpp vtu.cpp
    pta/definition.cpp pta/reader.cpp pta/writer.cpp pta/face_region_parser.cpp)
target_link_libraries(mhs_app_support PUBLIC mhs_metahotspot PRIVATE tinyxml2::tinyxml2 mhs_options)
target_include_directories(mhs_app_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MHS_BUILD_APPS)
add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_app_support mhs_options)
mhs_stage_runtime(metahotspot)
if(APPLE)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
endif()

if(MHS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
