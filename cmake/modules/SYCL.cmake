# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# if(USE_SYCL)
#   message(STATUS "Build with SYCL support")
#   message(STATUS "FOUND CMAKE_PREFIX_PATH : " ${CMAKE_PREFIX_PATH})
#   message(STATUS "FOUND CMAKE_FRAMEWORK_PATH : " ${CMAKE_FRAMEWORK_PATH})
#   message(STATUS "FOUND CMAKE_APPBUNDLE_PATH : " ${CMAKE_APPBUNDLE_PATH})
#   message(STATUS "FOUND PATH : " ${PATH})
#   tvm_file_glob(GLOB RUNTIME_SYCL_SRCS src/runtime/sycl/*.cc)
#   list(APPEND SYCL_RUNTIME_SRCS ${RUNTIME_SYCL_SRCS})
#   if($ENV{SYCL_ROOT_DIR})
#     include_directories(BEFORE SYSTEM $ENV{SYCL_ROOT_DIR}/include/sycl/)
#     list(APPEND TVM_RUNTIME_LINKER_LIBS $ENV{SYCL_ROOT_DIR}/lib/)
#   else()
#     # set(SYCL_DIR "/home/wzy/sycl_workspace/build-cuda")
#     set(SYCL_DIR "/home/wzy/sycl_workspace/tvm-sycl-build-cuda")
#     message(STATUS "FOUND SYCL_DIR : " ${SYCL_DIR})
#     include_directories(BEFORE SYSTEM ${SYCL_DIR}/include/sycl/)
#     list(APPEND TVM_RUNTIME_LINKER_LIBS ${SYCL_DIR}/lib/)
#   endif()
#   # include_directories(BEFORE SYSTEM $ENV{SYCL_ROOT_DIR}/include/sycl/)
#   # list(APPEND TVM_RUNTIME_LINKER_LIBS $ENV{SYCL_ROOT_DIR}/lib/)
# else()
#   list(APPEND COMPILER_SRCS src/target/opt/build_sycl_off.cc)
# endif(USE_SYCL)



# SYCL Module
find_sycl(${USE_SYCL})

if(SYCL_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(SYSTEM ${SYCL_INCLUDE_DIRS})
endif(SYCL_FOUND)

if(USE_SYCL)
  if (NOT SYCL_FOUND)
    find_package(SYCL REQUIRED)
  endif()
  message(STATUS "Build with SYCL support")
  tvm_file_glob(GLOB RUNTIME_SYCL_SRCS src/runtime/sycl/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${SYCL_LIBRARIES})

  if(DEFINED USE_SYCL_GTEST AND EXISTS ${USE_SYCL_GTEST})
    file_glob_append(RUNTIME_SYCL_SRCS
      "${CMAKE_SOURCE_DIR}/tests/cpp-runtime/sycl/*.cc"
    )
  endif()
  list(APPEND SYCL_RUNTIME_SRCS ${RUNTIME_SYCL_SRCS})
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_sycl_off.cc)
endif(USE_SYCL)

