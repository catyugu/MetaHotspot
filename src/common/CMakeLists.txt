mhs_add_library(common_lib
    SOURCES logger.cpp cli.cpp
    PUBLIC_LINKS spdlog
)

if(VERBOSE)
    target_compile_definitions(common_lib PRIVATE VERBOSE)
endif()
