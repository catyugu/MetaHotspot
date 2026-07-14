mhs_add_library(logger_lib
    SOURCES logger.cpp
    PUBLIC_LINKS spdlog
)

if(VERBOSE)
    target_compile_definitions(logger_lib PRIVATE VERBOSE)
endif()
