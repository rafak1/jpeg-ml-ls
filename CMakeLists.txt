cmake_minimum_required(VERSION 3.20)
project(jpegls_ai VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Library target
add_library(jpeglsai STATIC src/jpegls_ai.cpp)
target_include_directories(jpeglsai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Executable target
add_executable(jpegls_app src/main.cpp)
target_link_libraries(jpegls_app PRIVATE jpeglsai)
