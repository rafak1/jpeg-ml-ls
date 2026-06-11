cmake_minimum_required(VERSION 3.20)
project(jpegls_ai VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mavx2")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are forbidden! \n"
            "Zabraniam budowania projektu bezpośrednio w folderze ze źródłami.\n"
            "Usuń plik CMakeCache.txt oraz folder CMakeFiles, a następnie użyj osobnego folderu (np. cmake-build-debug).")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------------
# Fetch Dependencies Automatically
# -----------------------------------------------------------------------------
include(FetchContent)

# --- CharLS ---
set(BUILD_TESTING OFF CACHE BOOL "Disable tests for CharLS" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build CharLS as a static library" FORCE)

FetchContent_Declare(
        charls
        GIT_REPOSITORY https://github.com/team-charls/charls.git
        GIT_TAG        2.4.3
)
FetchContent_MakeAvailable(charls)

# --- STB Image ---
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# --- Dlib ---
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable Dlib GUI" FORCE)
set(DLIB_USE_CUDA OFF CACHE BOOL "Disable CUDA in Dlib" FORCE)

set(DLIB_JPEG_SUPPORT OFF CACHE BOOL "Disable JPEG" FORCE)
set(DLIB_PNG_SUPPORT OFF CACHE BOOL "Disable PNG" FORCE)
set(DLIB_GIF_SUPPORT OFF CACHE BOOL "Disable GIF" FORCE)
set(DLIB_WEBP_SUPPORT OFF CACHE BOOL "Disable WEBP" FORCE)
set(DLIB_JXL_SUPPORT OFF CACHE BOOL "Disable JXL" FORCE)

set(DLIB_USE_BLAS OFF CACHE BOOL "Disable BLAS" FORCE)
set(DLIB_USE_LAPACK OFF CACHE BOOL "Disable LAPACK" FORCE)

FetchContent_Declare(
        dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v20.0
)
FetchContent_MakeAvailable(dlib)
# -----------------------------------------------------------------------------

# Library target
add_library(jpeglsai STATIC
        src/jpegls_ai.cpp
        src/loco_i.cpp
        src/bit_stream.cpp
        src/ai/2_layer_chooser_ai_logic.cpp
        src/ai/2_layer_chooser_with_all_ai_logic.cpp
        src/ai/3_layer_chooser_with_all_ai_logic.cpp
        src/ai/cnn_chooser_with_all_ai_logic.cpp
        src/ai/3_layer_chooser_ai_logic.cpp
        src/ai/cnn_chooser_ai_logic.cpp
        src/utils/training_utils.cpp
        src/utils/benchmark_utils.cpp)

target_include_directories(jpeglsai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${stb_SOURCE_DIR})
target_link_libraries(jpeglsai PUBLIC dlib::dlib charls)

# Executable target
add_executable(jpegls_app src/main.cpp)
target_link_libraries(jpegls_app PRIVATE jpeglsai charls)

# Training executables
add_executable(train_2_layer_fc_predictor src/train_2_layer_fc_logic_model.cpp)
target_link_libraries(train_2_layer_fc_predictor PRIVATE jpeglsai charls dlib::dlib)

add_executable(train_3_layer_fc_predictor src/train_3_layer_fc_logic_model.cpp)
target_link_libraries(train_3_layer_fc_predictor PRIVATE jpeglsai charls dlib::dlib)

add_executable(train_cnn_predictor src/train_cnn_predictor_logic_model.cpp)
target_link_libraries(train_cnn_predictor PRIVATE jpeglsai charls dlib::dlib)

add_executable(train_2_layer_fc_predictor_for_all src/train_2_layer_chooser_with_all_ai_logic.cpp)
target_link_libraries(train_2_layer_fc_predictor PRIVATE jpeglsai charls dlib::dlib)

add_executable(train_3_layer_fc_predictor_for_all src/train_3_layer_chooser_with_all_ai_logic.cpp)
target_link_libraries(train_3_layer_fc_predictor PRIVATE jpeglsai charls dlib::dlib)

add_executable(train_cnn_predictor_for_all src/train_cnn_chooser_with_all_ai_logic.cpp)
target_link_libraries(train_cnn_predictor PRIVATE jpeglsai charls dlib::dlib)

# Wrapper to train and run benchmarks
add_executable(train_and_benchmark src/train_and_benchmark.cpp)
target_link_libraries(train_and_benchmark PRIVATE jpeglsai charls dlib::dlib)