cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(lumacam)

find_package(Geant4 10.0 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# Automatically include all .cc files from src/ and .hh files from include/
file(GLOB SOURCES src/*.cc)
file(GLOB HEADERS include/*.hh)

add_executable(lumacam ${SOURCES} ${HEADERS})
target_include_directories(lumacam PRIVATE include)
target_link_libraries(lumacam ${Geant4_LIBRARIES})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
