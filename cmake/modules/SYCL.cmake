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

if(USE_SYCL)
  message(STATUS "Build with SYCL support")
  tvm_file_glob(GLOB RUNTIME_SYCL_SRCS src/runtime/sycl/*.cc)
  list(APPEND SYCL_RUNTIME_SRCS ${RUNTIME_SYCL_SRCS})

  include_directories(BEFORE SYSTEM ${USE_SYCL}/include/sycl/)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${USE_SYCL}/lib/)
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_sycl_off.cc)
endif(USE_SYCL)
