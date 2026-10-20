# DTDM reproduction driver. Links the library's single target plus the
# third-party libraries its static archive needs at final link.
add_executable(mhs_dtdm
    dtdm.cpp
    mesh.cpp
    main.cpp
)

target_link_libraries(mhs_dtdm PRIVATE
    mhs::metahotspot
    Eigen3::Eigen
    TBB::tbb
)
target_compile_features(mhs_dtdm PRIVATE cxx_std_20)

# oneTBB is linked as a shared library on Windows, so the executable needs it
# next to itself to start.
mhs_copy_runtime_dlls(mhs_dtdm)

