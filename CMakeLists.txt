cmake_minimum_required(VERSION 3.11)
project(nano-nn VERSION 1.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

set(SOURCES
    src/main.c
    src/dataset.c
    src/neuralnetwork.c
)

set(HEADERS
    src/dataset.h
    src/core.h
    src/stb_image_resize2.h
    src/stb_image.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE m)