cmake_minimum_required(VERSION 3.11)
project(nano-nn VERSION 1.0)
include(GNUInstallDirs)

add_executable(${PROJECT_NAME} main.c main.h)

target_link_libraries(${PROJECT_NAME} PRIVATE m)
target_compile_options(nano-nn PUBLIC -lm)