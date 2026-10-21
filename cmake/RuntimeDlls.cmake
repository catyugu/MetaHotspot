# Runtime deployment for executables and shared libraries.
#
# $<TARGET_RUNTIME_DLLS:> is empty on non-DLL platforms, so the copy needs no
# platform guard, and it already covers every imported shared target in the link
# closure (oneTBB, OpenMP, oneMKL).

function(mhs_stage_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${target}>"
        $<TARGET_RUNTIME_DLLS:${target}>
        COMMAND_EXPAND_LISTS
    )
endfunction()
