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

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/delayomp)
if(TARGET mhs_delayomp)
    target_link_libraries(mhs_c_api PRIVATE mhs_delayomp)
endif()

target_compile_definitions(mhs_c_api PUBLIC MHS_BUILD_SHARED)
add_library(mhs::c_api ALIAS mhs_c_api)
# Every symbol is exported by default on ELF/Mach-O; hiding them makes the
# MHS_API attribute on the public header the only thing that escapes.
set_target_properties(mhs_c_api PROPERTIES
    EXPORT_NAME c_api
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(APPLE)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_property(TARGET mhs_c_api PROPERTY INSTALL_RPATH "$ORIGIN")
endif()
mhs_stage_runtime(mhs_c_api)

# scikit-build-core wheel: the Python package is a ctypes wrapper, so the
# library is installed into the package directory and the runtime libraries it
# needs travel with it (nothing guarantees oneTBB or an OpenMP runtime exists on
# a machine that only runs pip). System libraries stay with the system. Both
# rules sit in the "python" install component, the only one a wheel installs, so
# the headers and CMake packages of the CPM dependencies stay out of it.
if(SKBUILD)
    install(TARGETS mhs_c_api
        RUNTIME DESTINATION metahotspot
        LIBRARY DESTINATION metahotspot
        COMPONENT python)
    install(CODE [[
        # oneTBB is built by CPM and never installed, so its build directory has
        # to be a search location; Homebrew's OpenMP is outside the default
        # paths and announces itself through OpenMP_ROOT.
        set(_mhs_wheel_lookup "$<TARGET_FILE_DIR:mhs_c_api>" "$<TARGET_FILE_DIR:TBB::tbb>")
        if(DEFINED ENV{OpenMP_ROOT})
            list(APPEND _mhs_wheel_lookup "$ENV{OpenMP_ROOT}/lib")
        endif()
        file(GET_RUNTIME_DEPENDENCIES
            LIBRARIES "$<TARGET_FILE:mhs_c_api>"
            DIRECTORIES ${_mhs_wheel_lookup}
            RESOLVED_DEPENDENCIES_VAR _mhs_wheel_deps
            UNRESOLVED_DEPENDENCIES_VAR _mhs_wheel_unresolved
            PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
            POST_EXCLUDE_REGEXES
                ".*[Ss][Yy][Ss][Tt][Ee][Mm]32.*"
                ".*(^|/)lib(c|m|dl|rt|pthread|gcc_s|stdc\\+\\+)\\.so.*"
                ".*(^|/)ld-linux.*"
                "^/usr/lib/.*"
                "^/System/.*")
        foreach(_mhs_dep IN LISTS _mhs_wheel_deps)
            file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/metahotspot"
                TYPE SHARED_LIBRARY FILES "${_mhs_dep}")
        endforeach()
        if(_mhs_wheel_unresolved)
            message(WARNING "Unresolved runtime dependencies: ${_mhs_wheel_unresolved}")
        endif()
    ]] COMPONENT python)
endif()
