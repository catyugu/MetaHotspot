# Compiler options for strict build
if(MSVC)
    add_compile_options(/W4 /WX /bigobj /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-language-extension-token -Wno-microsoft-enum-value)
endif()
