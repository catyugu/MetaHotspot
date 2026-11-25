add_library(mhs_erom_extraction STATIC
    prepare.cpp
    extract.cpp
    basis.cpp
    spectrum.cpp
    compression.cpp
    amg_solver.cpp
    writer.cpp
)
add_library(mhs::erom_extraction ALIAS mhs_erom_extraction)
set_target_properties(mhs_erom_extraction PROPERTIES EXPORT_NAME erom_extraction)
target_link_libraries(mhs_erom_extraction
    PUBLIC mhs_model
    PRIVATE mhs_compiler mhs_solver mhs_io OpenMP::OpenMP_CXX $<BUILD_INTERFACE:mhs_tinyxml2_headers>
    $<BUILD_INTERFACE:mhs_options> $<BUILD_INTERFACE:Eigen3::Eigen> $<BUILD_INTERFACE:amgcl::amgcl>)
target_include_directories(mhs_erom_extraction
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/metahotspot>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/metahotspot>
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(mhs_erom_extraction PRIVATE AMGCL_NO_BOOST)
set_source_files_properties(amg_solver.cpp PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/wd4701>")

add_executable(metahotspot-erom main.cpp input.cpp)
target_link_libraries(metahotspot-erom PRIVATE mhs::erom_extraction mhs_pta_model_io mhs_tinyxml2_headers mhs_options)
mhs_stage_runtime(metahotspot-erom)
if(APPLE)
    set_property(TARGET metahotspot-erom PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot-erom PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(MHS_INSTALL)
    install(DIRECTORY include/metahotspot DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(MHS_BUILD_TESTS)
    add_executable(mhs_erom_check tests/check_embed.cpp)
    target_link_libraries(mhs_erom_check PRIVATE mhs::erom_extraction mhs::compiler mhs::solver mhs::io mhs_options)
    mhs_stage_runtime(mhs_erom_check)
    add_test(NAME mhs.erom.embed COMMAND mhs_erom_check --self-test "${CMAKE_CURRENT_BINARY_DIR}/erom-check.xml")
    set_tests_properties(mhs.erom.embed PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2")
endif()
