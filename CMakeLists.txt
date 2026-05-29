cmake_minimum_required(VERSION 3.20)
project(jpegls_ai VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Explicitly tell CMake where the headers are located for the entire project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------------
# Fetch CharLS Dependency Automatically
# -----------------------------------------------------------------------------
include(FetchContent)

# Disable CharLS tests and shared library generation to speed up build time
# and integrate cleanly into your project
set(BUILD_TESTING OFF CACHE BOOL "Disable tests for CharLS" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build CharLS as a static library" FORCE)

FetchContent_Declare(
        charls
        GIT_REPOSITORY https://github.com/team-charls/charls.git
        GIT_TAG        2.4.3 # Using a stable release tag
)
FetchContent_MakeAvailable(charls)
# -----------------------------------------------------------------------------

# Library target
add_library(jpeglsai STATIC src/jpegls_ai.cpp src/loco_i.cpp src/bit_stream.cpp)
target_include_directories(jpeglsai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable target
add_executable(jpegls_app src/main.cpp)

# Link your internal library AND the automatically downloaded CharLS library
target_link_libraries(jpegls_app PRIVATE jpeglsai charls)