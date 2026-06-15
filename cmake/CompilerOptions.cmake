# Project-wide warning/standard settings.
#
# These are applied ONLY to MetaHotspot's own targets, never to third-party
# dependencies brought in via CPM. Use mhs_set_strict_warnings() on each
# project target; do not call add_compile_options() at file scope.

function(mhs_set_strict_warnings TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /W4 /WX /bigobj /permissive- /utf-8
        )
    else()
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
            -Wno-language-extension-token
            -Wno-microsoft-enum-value
        )
    endif()
endfunction()
