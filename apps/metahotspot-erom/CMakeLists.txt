add_executable(metahotspot-erom
    main.cpp
    prepare.cpp
    extract.cpp
    basis.cpp
    spectrum.cpp
    compression.cpp
    amg_solver.cpp
    writer.cpp
)
target_link_libraries(metahotspot-erom PRIVATE
    metahotspot_frontend mhs_options Eigen3::Eigen amgcl::amgcl OpenMP::OpenMP_CXX)
target_include_directories(metahotspot-erom PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(metahotspot-erom PRIVATE AMGCL_NO_BOOST)
set_source_files_properties(amg_solver.cpp PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/wd4701>")
mhs_stage_runtime(metahotspot-erom)
if(APPLE)
    set_property(TARGET metahotspot-erom PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot-erom PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(MHS_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(mhs_erom_check tests/check_embed.cpp)
    target_link_libraries(mhs_erom_check PRIVATE metahotspot_frontend mhs_options)
    mhs_stage_runtime(mhs_erom_check)
    add_test(NAME mhs.erom.ExtractionRoundTrip
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/round_trip.py"
        $<TARGET_FILE:metahotspot-erom> $<TARGET_FILE:mhs_erom_check>)
endif()
