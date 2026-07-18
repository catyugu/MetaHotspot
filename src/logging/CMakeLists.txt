mhs_add_library(mhs_logging
    SOURCES logger.cpp
    PUBLIC_LINKS spdlog
)

if(VERBOSE)
    target_compile_definitions(mhs_logging PRIVATE VERBOSE)
endif()
