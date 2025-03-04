cmake_minimum_required(VERSION 3.16)

project(SAFETENSORS_CPP CXX C)

set(BUILD_TARGET_C "c-safetensors")

option(SAFETENSORS_CPP_BUILD_C_API "Build C API?" ON)
option(SAFETENSORS_CPP_BUILD_EXAMPLES "Build examples" ON)

# Disable C++ exception by default.
option(SAFETENSORS_CPP_CXX_EXCEPTIONS "Enable C++ exception(disable by default)" OFF)

set(SAFETENSORS_CPP_SOURCES
  safetensors.cc)

# compile with c++ for libsafetensors-c.a
set(SAFETENSORS_C_SOURCES
  safetensors-c.cc)

add_library(safetensors_cpp ${SAFETENSORS_CPP_SOURCES})

if (SAFETENSORS_CPP_BUILD_C_API)
  add_library(safetensors_c ${SAFETENSORS_C_SOURCES})
endif()

if(NOT SAFETENSORS_CPP_CXX_EXCEPTIONS)
  if(MSVC)
    # TODO: disable exception reliably
    #target_compile_options(safetensors_cpp PUBLIC /EHs-c-)
  else()
    target_compile_options(safetensors_cpp PUBLIC -fno-exceptions)
  endif()
endif()


if (SAFETENSORS_CPP_BUILD_EXAMPLES)
  add_executable(example example.cc)
  target_compile_definitions(example PRIVATE "SAFETENSORS_CPP_NO_IMPLEMENTATION")
  target_link_libraries(example safetensors_cpp)

  add_executable(serialize_example serialize-example.cc)
  target_compile_definitions(serialize_example PRIVATE "SAFETENSORS_CPP_NO_IMPLEMENTATION")
  target_link_libraries(serialize_example safetensors_cpp)

  if (SAFETENSORS_CPP_BUILD_C_API)
    add_executable(example-c example-c.c)
    target_compile_definitions(example-c PRIVATE "SAFETENSORS_C_NO_IMPLEMENTATION")
    target_link_libraries(example-c safetensors_c)
  endif ()
endif ()


install(TARGETS safetensors_cpp EXPORT safetensors-cpp
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)
set(install_headers
  safetensors.hh
)
if (SAFETENSORS_CPP_BUILD_C_API)
  list(APPEND install_headers
    safetensors-c.h
  )
endif (SAFETENSORS_CPP_BUILD_C_API)
install(FILES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${install_headers}
  DESTINATION include
)
# export full library interface
install(EXPORT safetensors-cpp
  FILE safetensors-cpp.cmake
  NAMESPACE safetensors-cpp::
  DESTINATION lib/cmake/safetensors-cpp
)
