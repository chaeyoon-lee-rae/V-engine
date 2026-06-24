cmake_minimum_required (VERSION 3.29)

set (TEMPLATE_NAME "VulkanTutorial")

project (${TEMPLATE_NAME})

# 모든 타깃(vulkan/std 모듈 + exe)을 UTF-8로 컴파일.
# - MSVC가 한글 주석을 CP949로 오독해 구문 에러 내는 것 방지
# - 전역 적용이라 모듈 producer/consumer의 _UTF8 환경이 일치 → 모듈 불일치 경고 방지
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS " /utf-8")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4.335 REQUIRED)
find_package(tinyobjloader REQUIRED)
find_path(TINYGLTF_INCLUDE_DIR tiny_gltf.h REQUIRED)
find_package(KTX REQUIRED)

# Add eva backend
set(EVA_ENABLE_WINDOW ON CACHE BOOL "Enable window support for eva" FORCE)
set(EVA_ENABLE_RAYTRACING ON CACHE BOOL "Enable raytracing support for eva" FORCE)
add_subdirectory(external/eva)

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

set(TEMPLATE_DIR ${CMAKE_BINARY_DIR}/${TEMPLATE_NAME})

add_executable (${TEMPLATE_NAME} legacy/main.cpp)
set_target_properties (${TEMPLATE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEMPLATE_DIR})
set_target_properties (${TEMPLATE_NAME} PROPERTIES CXX_STANDARD 23)
target_link_libraries (${TEMPLATE_NAME} eva glfw)
target_include_directories (${TEMPLATE_NAME} PRIVATE ${STB_INCLUDEDIR} ${TINYGLTF_INCLUDE_DIR})

# Define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS to treat VK_ERROR_OUT_OF_DATE_KHR as a success code
target_compile_definitions(${TEMPLATE_NAME} PRIVATE 
    "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS"
    "VULKAN_HPP_NO_STRUCT_CONSTRUCTORS"
    "VULKAN_HPP_NO_UNION_CONSTRUCTORS"
)

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set_target_properties(${TEMPLATE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endif()

# Slang shader
set (SHADER_TARGET ${TEMPLATE_NAME}Shader)
set (SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

set(SLANG_SRC ${SHADERS_DIR}/raytracing.slang)

set(RT_TARGETS 
  "rgenMain,raytracing.rgen.spv"
  "missMain,raytracing.rmiss.spv"
  "chitMain,raytracing.rchit.spv"
)

set(SPV_OUTPUTS "")
foreach(TARGET_INFO_STR IN LISTS RT_TARGETS)
  string(REPLACE "," ";" TARGET_INFO "${TARGET_INFO_STR}")
  list(GET TARGET_INFO 0 ENTRY_POINT)
  list(GET TARGET_INFO 1 OUTPUT_NAME)
  set(SPV_FILE ${SHADERS_DIR}/${OUTPUT_NAME})
  list(APPEND SPV_OUTPUTS ${SPV_FILE})
  add_custom_command(
    OUTPUT  ${SPV_FILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND ${SLANGC_EXECUTABLE} ${SLANG_SRC} -target spirv -profile spirv_1_4 -emit-spirv-directly -entry ${ENTRY_POINT} -o ${SPV_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${SLANG_SRC}
    COMMENT "Compiling Slang Shader: ${ENTRY_POINT} -> ${OUTPUT_NAME}"
    VERBATIM
  )
endforeach()

add_custom_target(${SHADER_TARGET} DEPENDS ${SPV_OUTPUTS})
add_dependencies(${TEMPLATE_NAME} ${SHADER_TARGET})

# Assets
# set (ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets/")
# file (COPY ${ASSET_DIR}/viking_room.obj DESTINATION ${TEMPLATE_DIR}/models)
# file (COPY ${ASSET_DIR}/viking_room.png DESTINATION ${TEMPLATE_DIR}/textures)