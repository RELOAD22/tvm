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

#######################################################
# Enhanced version of find SYCL.
#
# Usage:
#   find_sycl(${USE_SYCL})
#
# - When USE_SYCL=ON, use auto search
# - When USE_SYCL=/path/to/sycl-sdk-path, use the sdk.
#   Can be useful when cross compiling and cannot rely on
#   CMake to provide the correct library as part of the
#   requested toolchain.
#
# Provide variables:
#
# - SYCL_FOUND
# - SYCL_INCLUDE_DIRS
# - SYCL_LIBRARIES
#

macro(find_sycl use_sycl)
  set(__use_sycl ${use_sycl})
  if(IS_DIRECTORY ${__use_sycl})
    set(__sycl_sdk ${__use_sycl})
    message(STATUS "Custom SYCL SDK PATH=" ${__use_sycl})
   elseif(IS_DIRECTORY $ENV{SYCL_SDK})
     set(__sycl_sdk $ENV{SYCL_SDK})
   else()
     set(__sycl_sdk "")
   endif()

   if(__sycl_sdk)
     set(SYCL_INCLUDE_DIRS ${__sycl_sdk}/include)
     message(STATUS "SYCL_INCLUDE_DIRS=" ${SYCL_INCLUDE_DIRS})
     if (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY STREQUAL "ONLY")
       set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
     endif()
     # we are in the section dedicated to the explicit pointing of SYCL SDK path, we must not
     # look for the SYCL library by default path, but should be limited by provided SDK
     set(CMAKE_BUILD_RPATH "${__sycl_sdk}/lib;${CMAKE_BUILD_RPATH}")
     message(STATUS "set sycl lib directory in cmake build rpath : ${__sycl_sdk}/lib ")
     link_directories(${__sycl_sdk}/lib)
     message(STATUS "set sycl link_directories : ${__sycl_sdk}/lib ")
     find_library(SYCL_LIBRARIES NAMES sycl NO_DEFAULT_PATH PATHS ${__sycl_sdk}/lib ${__sycl_sdk}/lib64 ${__sycl_sdk}/lib/x64/)
     message(STATUS "SYCL_LIBRARIES=" ${SYCL_LIBRARIES})
     if(SYCL_LIBRARIES)
       set(SYCL_FOUND TRUE)
     endif()
   endif(__sycl_sdk)

   # No user provided SYCL include/libs found
   if(NOT SYCL_FOUND)
     if(${__use_sycl} MATCHES ${IS_TRUE_PATTERN})
       find_package(SYCL QUIET)
     endif()
   endif()

  if(SYCL_FOUND)
    message(STATUS "SYCL_INCLUDE_DIRS=" ${SYCL_INCLUDE_DIRS})
    message(STATUS "SYCL_LIBRARIES=" ${SYCL_LIBRARIES})
  endif(SYCL_FOUND)
endmacro(find_sycl)