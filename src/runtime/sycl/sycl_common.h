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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>

/* There are many OpenCL platforms that do not yet support OpenCL 2.0,
 * hence we use 1.2 APIs, some of which are now deprecated.  In order
 * to turn off the deprecation warnings (elevated to errors by
 * -Werror) we explicitly disable the 1.2 deprecation warnings.
 *
 * At the point TVM supports minimum version 2.0, we can remove this
 * define.
 */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

/* Newer releases of OpenCL header files (after May 2018) work with
 * any OpenCL version, with an application's target version
 * specified. Setting the target version disables APIs from after that
 * version, and sets appropriate USE_DEPRECATED macros.  The above
 * macro for CL_USE_DEPRECATED_OPENCL_1_2_APIS is still needed in case
 * we are compiling against the earlier version-specific OpenCL header
 * files.  This also allows us to expose the OpenCL version through
 * tvm.runtime.Device.
 */
// #define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <sycl/sycl.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../texture.h"
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
    }catch(sycl::exception const& e){                   \
      SYCL_CHECK_ERROR(e.code())                              \
      std::cout << "Caught synchronous SYCL exception:\n"     \
              << e.what() << std::endl;                       \
    }                                                         \
  }

static_assert(sizeof(cl_mem) == sizeof(void*), "Required to store cl_mem inside void*");


class SYCLThreadEntry;

/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace : public DeviceAPI {
 public:
  // type key
  std::string type_key;
  // global platform id
  // global platform
  std::vector<sycl::platform> platforms;
  // global platform name
  std::vector<std::string> platform_names;
  // global context of this process
  std::vector<sycl::context> contexts;
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
  // Number of registered kernels
  // Used to register kernel into the workspace.
  size_t num_registered_kernels{0};
  // The version counter, used
  size_t timestamp{0};
  // Ids that are freed by kernels.
  std::vector<size_t> free_kernel_ids;
  // the mutex for initialization
  std::mutex mu;


  // destructor
  ~SYCLWorkspace() {
    LOG(WARNING) << "todo, not support now";
    for(auto queue : queues){
      SYCL_CALL(queue.wait_and_throw());
    }
  }
  // Initialzie the device.
  void Init(const std::string& type_key, const std::string& device_type,
            const std::string& platform_name = "");
  virtual void Init() { Init("sycl", "gpu"); }
  // Check whether the context is SYCL or not.
  virtual bool IsSYCLDevice(Device dev) { return dev.device_type == kDLSYCL; }

  // get the queue of the device
  sycl::queue GetQueue(Device dev) {
    VLOG(1) << "sycl get queue device type " << dev.device_type << std::endl; 
    // std::cout << "sycl get queue device type " << dev.device_type << std::endl; 
    ICHECK(IsSYCLDevice(dev) );
    this->Init();
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid SYCL device_id=" << dev.device_id;
    return queues[dev.device_id];
  }
  // get the event queue of the context
  std::vector<sycl::event>& GetEventQueue(Device dev) {
    VLOG(1) << "sycl get event queue device type " << dev.device_type << std::endl; 
    // std::cout << "sycl get event queue device type " << dev.device_type << std::endl; 
    ICHECK(IsSYCLDevice(dev));
    this->Init();
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid SYCL device_id=" << dev.device_id;
    return events[dev.device_id];
  }
  // is current syclQueue in profiling mode
  bool IsProfiling(Device dev) {
    sycl::queue queue = GetQueue(dev);
    return queue.has_property<sycl::property::queue::enable_profiling>();
  }

  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final;
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope = NullOpt) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  // Texture (image2d_t) alloca APIs
  cl_mem AllocTexture(Device dev, size_t width, size_t height, DLDataType type_hint);
  void* AllocTextureWorkspace(Device dev, size_t width, size_t height, DLDataType type_hint);
  void FreeTextureWorkspace(Device dev, void* data);

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
  // The kernel entry and version.
  struct KTEntry {
    // not support
    std::string so_path;
    void * so_handler = nullptr;
    // timestamp used to recognize stale kernel
    size_t version{0};
  };
  /*! \brief The current device */
  Device device;
  /*! \brief The thread-local kernel table */
  std::vector<KTEntry> kernel_table;
  /*! \brief workspace pool */
  WorkspacePool pool;
  /*! \brief texture pool */
  TexturePool texture_pool;
  // constructor
  SYCLThreadEntry(DLDeviceType device_type, DeviceAPI* device_api)
      : pool(device_type, device_api), texture_pool(device_type, device_api) {
    device.device_id = 0;
    device.device_type = device_type;
  }
  SYCLThreadEntry() : SYCLThreadEntry(DLDeviceType(kDLSYCL), SYCLWorkspace::Global()) {}

  // get the global workspace
  static SYCLThreadEntry* ThreadLocal();
};

/*! \brief SYCL runtime buffer structure with tracked memory layout
    TODO(tvm-team): Uncouple use of storage scope and data layout by using the transform_layout
    schedule primitive to express the desired texture layout. This will require supporting Nd
    indices in BufferLoad and BufferStore in CodegenSYCL, and ensuring Nd allocations for
    texture are correctly routed to the AllocateTexture packed function in the SYCL DeviceAPI.
*/
struct syclBufferDescriptor {
  enum class MemoryLayout {
    /*! \brief One dimensional buffer in row-major layout*/
    kBuffer1D,
    /*! \brief Two dimensional texture w/ width = axis[-1]
     *          e.g. image2d[height=NCH, width=W]
     */
    kImage2DActivation,
    /*! \brief Two dimensional texture w/ height = axis[0]
     *         e.g. image2d[height=O, width=IHW]
     */
    kImage2DWeight,
    /*! \brief Two dimensional texture w/ height = axis[1]
     *         e.g. image2d[height=NH, width=WC]
     */
    kImage2DNHWC,
  };
  syclBufferDescriptor() = default;
  explicit syclBufferDescriptor(Optional<String> scope) : layout(MemoryLayoutFromScope(scope)) {}
  static MemoryLayout MemoryLayoutFromScope(Optional<String> mem_scope);
  static String ScopeFromMemoryLayout(MemoryLayout mem_scope);

  cl_mem buffer{nullptr};
  MemoryLayout layout{MemoryLayout::kBuffer1D};
};
}  // namespace syclT

// Module to support thread-safe multi-device execution.
// SYCL runtime is a bit tricky because clSetKernelArg is not thread-safe
// To make the call thread-safe, we create a thread-local kernel table
// and lazily install new kernels into the kernel table when the kernel is called.
// The kernels are recycled when the module get destructed.
class SYCLModuleNode : public ModuleNode {
 public:
  // Kernel table reference entry.
  struct KTRefEntry {
    size_t kernel_id;
    size_t version;
  };
  explicit SYCLModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {}
  // destructor
  ~SYCLModuleNode();

  /*!
   * \brief Get the global workspace
   */
  virtual syclT::SYCLWorkspace* GetGlobalWorkspace();

  const char* type_key() const final { return workspace_->type_key.c_str(); }

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
  // Module local mutex
  std::mutex build_lock_;
  // The SYCL source.
  std::string source_;
  // kernel id cache. Mapping from primitive name to KTRefEntry.
  std::unordered_map<std::string, KTRefEntry> kid_map_;
  // kernels build so far.
  std::vector<cl_kernel> kernels_;
  // parsed kernel data, Mapping from primitive name to kernel code.
  std::unordered_map<std::string, std::string> parsed_kernels_;
  // share library handler
  void * so_handler_ = nullptr;
};

/*! \brief SYCL timer node */
// TODO

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SYCL_SYCL_COMMON_H_