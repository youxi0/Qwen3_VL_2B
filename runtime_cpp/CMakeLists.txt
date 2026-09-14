cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(qwen3_vl_runtime VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# set(变量名 值 CACHE 类型 说明)
set(EDGELLM_SOURCE_DIR "/home/jetson/TensorRT-Edge-LLM" CACHE PATH
    "TensorRT Edge-LLM source directory")
set(EDGELLM_BUILD_DIR "${EDGELLM_SOURCE_DIR}/build-v0101" CACHE PATH
    "TensorRT Edge-LLM build directory")
set(TRT_PACKAGE_DIR "/usr" CACHE PATH "TensorRT package root")

find_package(CUDAToolkit REQUIRED)

find_path(
  TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${TRT_PACKAGE_DIR}"
  PATH_SUFFIXES include include/aarch64-linux-gnu)
find_library(
  TENSORRT_LIBRARY
  NAMES nvinfer
  HINTS "${TRT_PACKAGE_DIR}"
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)
find_library(
  TENSORRT_ONNX_PARSER_LIBRARY
  NAMES nvonnxparser
  HINTS "${TRT_PACKAGE_DIR}"
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)

set(EDGELLM_CORE_LIBRARY "${EDGELLM_BUILD_DIR}/cpp/libedgellmCore.a")
set(EDGELLM_CUTEDSL_LIBRARY
    "${EDGELLM_SOURCE_DIR}/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a")

foreach(required_path IN ITEMS "${TENSORRT_INCLUDE_DIR}" "${TENSORRT_LIBRARY}"
                               "${TENSORRT_ONNX_PARSER_LIBRARY}"
                               "${EDGELLM_CORE_LIBRARY}"
                               "${EDGELLM_CUTEDSL_LIBRARY}")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "Required dependency not found: ${required_path}")
  endif()
endforeach()

add_library(edgellm_core STATIC IMPORTED GLOBAL)
set_target_properties(
  edgellm_core
  PROPERTIES IMPORTED_LOCATION "${EDGELLM_CORE_LIBRARY}"
             IMPORTED_LINK_INTERFACE_LANGUAGES "CXX;CUDA")

add_library(edgellm_cutedsl STATIC IMPORTED GLOBAL)
set_target_properties(
  edgellm_cutedsl
  PROPERTIES IMPORTED_LOCATION "${EDGELLM_CUTEDSL_LIBRARY}"
             IMPORTED_LINK_INTERFACE_LANGUAGES "CXX;CUDA")

add_library(qwen3_vl_runtime STATIC src/qwen3VlRuntime.cpp)
target_include_directories(
  qwen3_vl_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${EDGELLM_SOURCE_DIR}/cpp
          ${EDGELLM_SOURCE_DIR}/3rdParty/nlohmannJson/include
          ${EDGELLM_SOURCE_DIR}/3rdParty/stb
          ${EDGELLM_SOURCE_DIR}/3rdParty/miniaudio
          ${TENSORRT_INCLUDE_DIR})
target_compile_definitions(qwen3_vl_runtime PRIVATE TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
target_compile_options(qwen3_vl_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(
  qwen3_vl_runtime
  PUBLIC edgellm_core
  PRIVATE edgellm_cutedsl
          "${TENSORRT_LIBRARY}"
          "${TENSORRT_ONNX_PARSER_LIBRARY}"
          CUDA::cuda_driver
          CUDA::cudart
          ${CMAKE_DL_LIBS}
          rt
          pthread)
set_target_properties(
  qwen3_vl_runtime
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             POSITION_INDEPENDENT_CODE ON)

add_executable(qwen3_vl_cli src/main.cpp)
target_include_directories(
  qwen3_vl_cli
  PRIVATE ${EDGELLM_SOURCE_DIR}/3rdParty/nlohmannJson/include)
target_compile_options(qwen3_vl_cli PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(qwen3_vl_cli PRIVATE qwen3_vl_runtime)
target_link_options(qwen3_vl_cli PRIVATE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")
set_target_properties(qwen3_vl_cli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
