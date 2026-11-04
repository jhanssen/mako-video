cmake_minimum_required(VERSION 4.3)
project(mako_video LANGUAGES CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(FATAL_ERROR "mako-video targets Apple Silicon only")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    message(STATUS "CMAKE_BUILD_TYPE not set: using RelWithDebInfo")
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
# Warnings fail the build (C++ and Metal); cmake --compile-no-warning-error turns that off for one configure.
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/mako-common/CMakeLists.txt)
    message(FATAL_ERROR "external/mako-common is missing: git submodule update --init")
endif()
add_subdirectory(external/mako-common)

set(S ${CMAKE_CURRENT_SOURCE_DIR}/src)
mako_add_kernels(video_kernels SOURCES ${S}/kernels/video.metal)

add_library(mako_videogen STATIC ${S}/video/h3.cpp ${S}/video/video_vae.cpp ${S}/video/audio_vae.cpp
            ${S}/video/engine.cpp ${S}/video/mp4.mm)
target_include_directories(mako_videogen PUBLIC ${S})
target_link_libraries(mako_videogen PUBLIC mako_gpu mako_models mako_safetensors mako_text mako_log
                      PRIVATE glaze_headers "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo"
                      "-framework Foundation" "-framework ImageIO" "-framework CoreGraphics")
set_source_files_properties(${S}/video/mp4.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

add_executable(mako_video tools/mako_video.cpp)
target_link_libraries(mako_video PRIVATE mako_videogen mako_memory mako_app)

# Model-backed tests need the model's directory and the reference tensors; without them they report as skipped.
set(MAKO_H3 "$ENV{MAKO_H3}" CACHE PATH "MiniMax-H3 FL2VA directory (its original files) for its tests")
set(MAKO_H3_REF "$ENV{MAKO_H3_REF}" CACHE PATH "MiniMax-H3 reference tensors (tools/dump_h3_ref.py) for its tests")
enable_testing()
add_subdirectory(tests)
