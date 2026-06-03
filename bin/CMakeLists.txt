add_executable(metahotspot main.cpp)
target_link_libraries(metahotspot PRIVATE mhs_lib)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
)
add_custom_command(TARGET metahotspot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TBB::tbb>
    $<TARGET_FILE_DIR:metahotspot>
)
