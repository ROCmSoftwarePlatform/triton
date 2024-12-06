cmake_minimum_required(VERSION 3.20)
project(ck-gemm-runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

option(USE_STREAM_PIPELINE "use stream pipeline" OFF)
option(USE_INTERWAVE "use interwave scheduling" OFF)

if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
  else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
  endif()
endif()

if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which HIP has been installed")
  else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
  endif()
endif()

if(NOT DEFINED CK_PATH)
  if(NOT DEFINED ENV{CK_PATH})
      set(CK_PATH "/opt/rocm" CACHE PATH "Path to which CK has been installed")
  else()
      set(CK_PATH $ENV{CK_PATH} CACHE PATH "Path to which CK has been installed")
  endif()
endif()

set(CMAKE_PREFIX_PATH "${CK_PATH}" "${HIP_PATH}" "${ROCM_PATH}")

include(FetchContent)
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        v2.2.0
)
FetchContent_MakeAvailable(cli11)

find_package(hip REQUIRED)
message(STATUS "Found HIP executable at: ${HIP_BIN_INSTALL_DIR}")


find_package(composable_kernel 1.1.0 COMPONENTS device_gemm_operations CONFIG PATHS ${CK_PATH})
if (NOT composable_kernel_FOUND)
  message(FATAL_ERROR "faild to find composable kernels")
else()
  message(STATUS "CK PATH: ${CK_PATH}")
endif()

set(LIBS composable_kernel::device_gemm_operations hip::device)

add_library(kernel SHARED ${PROJECT_SOURCE_DIR}/kernel.cpp)
target_link_libraries(kernel PUBLIC ${LIBS})
target_compile_options(kernel PUBLIC --save-temps)

find_package(Threads REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC kernel CLI11::CLI11 Threads::Threads)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
