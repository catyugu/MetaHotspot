add_library(logger_lib STATIC logger.cpp)
target_link_libraries(logger_lib PUBLIC spdlog)
target_include_directories(logger_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)
