cmake_minimum_required(VERSION 3.11)
project(nano-nn VERSION 1.0)
include(GNUInstallDirs)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
add_executable(${PROJECT_NAME} src/main.c src/main.h)

target_link_libraries(${PROJECT_NAME} PRIVATE m)
target_compile_options(nano-nn PUBLIC -lm)