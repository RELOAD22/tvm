/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sycl_common.h
 * \brief SYCL common header
 */
#ifndef TVM_RUNTIME_SYCL_SYCL_COMMON_H_
#define TVM_RUNTIME_SYCL_SYCL_COMMON_H_

#include <CL/sycl.hpp>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>  //open("/dev/urandom", O_RDONLY);
#ifdef _WIN32
    #include <process.h>
#else
    #include <unistd.h>
#endif

#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace syclT {

inline const char* SYCLGetErrorString(std::error_code error_code) {
  sycl::errc error_code_value = static_cast<sycl::errc>(error_code.value());
  switch(error_code_value){
    case sycl::errc::success:
      return "SUCCESS";
    case sycl::errc::runtime:
      return "RUNTIME ERROR";
    case sycl::errc::kernel:
      return "KERNEL ERROR";
    case sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case sycl::errc::nd_range:
      return "NDRANGE ERROR";
    case sycl::errc::event:
      return "EVENT ERROR";
    case sycl::errc::kernel_argument:
      return "KERNEL ARGUMNET ERROR";
    case sycl::errc::build:
      return "BUILD ERROR";
    case sycl::errc::invalid:
      return "INVALID ERROR";
    case sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION";
    case sycl::errc::platform:
      return "PLATFORM ERROR";
    case sycl::errc::profiling:
      return "PROFILING ERROR";
    case sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case sycl::errc::kernel_not_supported:
      return "kERNEL NOT SUPPORTED";
    case sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";     
  }
}
#define SYCL_CHECK_ERROR(e) \
  { ICHECK(e == sycl::errc::success) << "SYCL Error, code=" << e << ": " << syclT::SYCLGetErrorString(e); }

/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CALL(func)                                       \
  {                                                           \
    try{                                                      \
      func;                                                   \
    }catch(const sycl::exception &e){                         \
      /*针对USM等sycl runtime error，终止当前进程*/             \
      if(e.code() == sycl::errc::runtime){                    \
        std::exit(0);                                         \
      }                                                       \
      SYCL_CHECK_ERROR(e.code())                              \
      std::cout<< "Caught synchronous SYCL exception:" <<e.what()<< std::endl;\
    }                                                         \
  }


class SYCL_LIB_COMPILER {
  public:
    SYCL_LIB_COMPILER(){}
    SYCL_LIB_COMPILER(int module_id){
      int pid = (int)getpid();
      std::string filename = prefix + "/sycl_" + std::to_string(pid) + "_" +std::to_string(module_id);
      //std::string filename = prefix + "/sycl_" + getUUID();
      source_file_path = filename + ".cc";
      shared_lib_path = filename + ".so";
      command = sycl_compiler +" "+sycl_flags +" "+ source_file_path +" -o "+shared_lib_path;
    }

    std::string sycl_compiler = SYCL_CXX_COMPILER;
    std::string sycl_flags = SYCL_FLAGS;
    std::string prefix = SYCL_TEMP_FOLDER;
    std::string source_file_path;
    std::string shared_lib_path;
    std::string command;
  private:
    std::string getUUID(unsigned int len = 5){
      std::string str;
      unsigned char* buffer = new unsigned char[len+1];
      char* uuid = new char [2*len+1];
      int fd = open("/dev/urandom", O_RDONLY);
      if (read(fd, buffer, len) == len) {
        for(int i = 0;i < len;i++)
            sprintf(uuid + i*2, "%02X",buffer[i]);
        uuid[2*len] = '\0';
        str = uuid;
        
      } else {
        printf("Error: GetUnique %d\n", __LINE__);
      }
      delete [] buffer;
      delete [] uuid;
      return str;
    }
};

class SYCLThreadEntry;

/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace : public DeviceAPI {
 public:
  // global platform
  sycl::platform platform;
  // global platform name
  std::string platform_name;
  // global context of this process
  sycl::context context;
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<sycl::device> devices;
  // the queues
  std::vector<sycl::queue> queues;
  // the events
  std::vector<std::vector<sycl::event>> events;
  // the mutex for initialization
  std::mutex mu;
  // the number of sycl_modules
  int module_num = 0;
  // destructor
  ~SYCLWorkspace() {
    for(auto queue : queues){
      SYCL_CALL(queue.wait_and_throw());
    }
  }
  // Initialzie the device.
  void Init(const std::string& device_type, const std::string& platform_name = "");
  virtual void Init() { Init("gpu"); }
  // Check whether the context is SYCL or not.
  virtual bool IsSYCLDevice(Device dev) { return dev.device_type == static_cast<DLDeviceType>(kDLSYCL); }
  // get the queue of the device
  sycl::queue GetQueue(Device dev);
  // get the event queue of the context
  std::vector<sycl::event>& GetEventQueue(Device dev);
  // is current syclQueue in profiling mode
  bool IsProfiling(Device dev);

  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;

  /*!
   * \brief Get the thread local ThreadEntry
   */
  virtual SYCLThreadEntry* GetThreadEntry();

  // get the global workspace
  static SYCLWorkspace* Global();

  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;  
                           
};

/*! \brief Thread local workspace */
class SYCLThreadEntry {
 public:
  /*! \brief The current device */
  Device device;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  SYCLThreadEntry(DLDeviceType device_type, DeviceAPI* device_api)
      : pool(device_type, device_api) {
    device.device_id = 0;
    device.device_type = device_type;
  }
  SYCLThreadEntry() : SYCLThreadEntry(DLDeviceType(kDLSYCL), SYCLWorkspace::Global()) {}

  // get the global workspace
  static SYCLThreadEntry* ThreadLocal();
};

}  // namespace syclT

// Module to support thread-safe multi-device execution.
// SYCL runtime is a bit tricky because clSetKernelArg is not thread-safe
// To make the call thread-safe, we create a thread-local kernel table
// and lazily install new kernels into the kernel table when the kernel is called.
// The kernels are recycled when the module get destructed.
class SYCLModuleNode : public ModuleNode {
 public:
  explicit SYCLModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
        workspace_ = GetGlobalWorkspace();
        workspace_->module_num++;           //the number of sycl_modules add 1
        lib_compiler = syclT::SYCL_LIB_COMPILER(workspace_->module_num);
      }
  // destructor
  ~SYCLModuleNode();

  /*!
   * \brief Get the global workspace
   */
  virtual syclT::SYCLWorkspace* GetGlobalWorkspace();

  const char* type_key() const final { return "sycl"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;
  void SaveToFile(const std::string& file_name, const std::string& format) final;
  void SaveToBinary(dmlc::Stream* stream) final;
  std::string GetSource(const std::string& format) final;
  // Initialize the programs
  void Init();

 private:
  // The workspace, need to keep reference to use it in destructor.
  // In case of static destruction order problem.
  syclT::SYCLWorkspace* workspace_;
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table. Mapping from primitive name to .
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The SYCL source.
  std::string source_;
  // parsed kernel data, Mapping from primitive name to kernel code.
  std::unordered_map<std::string, std::string> parsed_kernels_;
  //id of SYCLModuleNode, start from 1, to detemine unique filename of source code and share library
  syclT::SYCL_LIB_COMPILER lib_compiler;
  // share library handler
  void * so_handler_ = nullptr;
  // share library name
  std::string dynamic_library_name;
  // share library path
  static const std::string library_path;
};

/*! \brief SYCL timer node */
class SYCLTimerNode : public TimerNode {
 public:
  // Timer start
  virtual void Start(){
    this->start = std::chrono::steady_clock::now();
  }
  virtual void Stop(){
    this->end = std::chrono::steady_clock::now();
  }
  virtual int64_t SyncAndGetElapsedNanos(){ 
    return std::chrono::duration_cast<std::chrono::nanoseconds>(this->end - this->start).count();
  }
  virtual ~SYCLTimerNode() {
    
  }
  SYCLTimerNode() {
  }
  static constexpr const char* _type_key = "SYCLTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SYCLTimerNode, TimerNode);
 private:
  std::chrono::steady_clock::time_point start, end;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SYCL_SYCL_COMMON_H_
