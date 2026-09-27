cmake_minimum_required(VERSION 3.15)
project(g1_loco_bridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

find_library(UNITREE_SDK2_LIB NAMES unitree_sdk2 PATHS /usr/local/lib /usr/lib)
find_path(
  UNITREE_SDK2_INCLUDE
  NAMES unitree/robot/g1/loco/g1_loco_client.hpp
  PATHS /usr/local/include /usr/include
)

find_library(DDSC_LIB NAMES ddsc PATHS /usr/local/lib /usr/lib)
find_library(DDSCXX_LIB NAMES ddscxx PATHS /usr/local/lib /usr/lib)
find_path(
  DDSCXX_ROOT_INCLUDE
  NAMES dds/dds.hpp
  PATHS /usr/local/include/ddscxx /usr/include
)

if(NOT UNITREE_SDK2_LIB OR NOT UNITREE_SDK2_INCLUDE)
  message(FATAL_ERROR "A unitree_sdk2 installation with G1 LocoClient support is required")
endif()

if(NOT DDSC_LIB OR NOT DDSCXX_LIB OR NOT DDSCXX_ROOT_INCLUDE)
  message(FATAL_ERROR "CycloneDDS C/C++ libraries and headers are required")
endif()

python_add_library(
  _g1_loco_bridge
  MODULE
  src/py_binding.cpp
  src/g1_loco_controller.cpp
  WITH_SOABI
)

target_include_directories(
  _g1_loco_bridge
  PRIVATE
  ${UNITREE_SDK2_INCLUDE}
  ${DDSCXX_ROOT_INCLUDE}
)

target_link_libraries(
  _g1_loco_bridge
  PRIVATE
  pybind11::headers
  ${UNITREE_SDK2_LIB}
  ${DDSC_LIB}
  ${DDSCXX_LIB}
  ${CMAKE_DL_LIBS}
  Threads::Threads
)

set_target_properties(
  _g1_loco_bridge
  PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN:/usr/local/lib"
)

install(TARGETS _g1_loco_bridge DESTINATION g1_loco_bridge)
