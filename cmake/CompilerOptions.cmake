# Compiler options for strict build
if(MSVC)
    add_compile_options(/W4 /WX /wd4819)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()
