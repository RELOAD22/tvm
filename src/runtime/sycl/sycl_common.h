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
#ifndef TVM_RUNTIME_SYPI_ERROR_SYPI_ERROR_COMMON_H_
#define TVM_RUNTIME_SYPI_ERROR_SYPI_ERROR_COMMON_H_

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
#define PI_ERROR_USE_DEPRECATED_OPENPI_ERROR_1_2_APIS

/* Newer releases of OpenCL header files (after May 2018) work with
 * any OpenCL version, with an application's target version
 * specified. Setting the target version disables APIs from after that
 * version, and sets appropriate USE_DEPRECATED macros.  The above
 * macro for PI_ERROR_USE_DEPRECATED_OPENPI_ERROR_1_2_APIS is still needed in case
 * we are compiling against the earlier version-specific OpenCL header
 * files.  This also allows us to expose the OpenCL version through
 * tvm.runtime.Device.
 */
// #define PI_ERROR_TARGET_OPENPI_ERROR_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <CL/sycl.hpp>

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

static_assert(sizeof(cl_mem) == sizeof(void*), "Required to store cl_mem inside void*");

inline const char* SYCLGetErrorString(pi_result error) {
  switch (error) {
    case PI_SUCCESS:
      return "PI_SUCCESS";
    case PI_ERROR_DEVICE_NOT_FOUND:
      return "PI_ERROR_DEVICE_NOT_FOUND";
    case PI_ERROR_COMPILER_NOT_AVAILABLE:
      return "PI_ERROR_COMPILER_NOT_AVAILABLE";
    case PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
      return "PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
    case PI_ERROR_OUT_OF_RESOURCES:
      return "PI_ERROR_OUT_OF_RESOURCES";
    case PI_ERROR_OUT_OF_HOST_MEMORY:
      return "PI_ERROR_OUT_OF_HOST_MEMORY";
    case PI_ERROR_PROFILING_INFO_NOT_AVAILABLE:
      return "PI_ERROR_PROFILING_INFO_NOT_AVAILABLE";
    case PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
      return "PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
    case PI_ERROR_BUILD_PROGRAM_FAILURE:
      return "PI_ERROR_BUILD_PROGRAM_FAILURE";
    case PI_ERROR_INVALID_VALUE:
      return "PI_ERROR_INVALID_VALUE";
    case PI_ERROR_INVALID_PLATFORM:
      return "PI_ERROR_INVALID_PLATFORM";
    case PI_ERROR_INVALID_DEVICE:
      return "PI_ERROR_INVALID_DEVICE";
    case PI_ERROR_INVALID_CONTEXT:
      return "PI_ERROR_INVALID_CONTEXT";
    case PI_ERROR_INVALID_QUEUE_PROPERTIES:
      return "PI_ERROR_INVALID_QUEUE_PROPERTIES";
    case PI_ERROR_INVALID_MEM_OBJECT:
      return "PI_ERROR_INVALID_MEM_OBJECT";
    case PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case PI_ERROR_INVALID_IMAGE_SIZE:
      return "PI_ERROR_INVALID_IMAGE_SIZE";
    case PI_ERROR_INVALID_SAMPLER:
      return "PI_ERROR_INVALID_SAMPLER";
    case PI_ERROR_INVALID_BINARY:
      return "PI_ERROR_INVALID_BINARY";
    case PI_ERROR_INVALID_PROGRAM:
      return "PI_ERROR_INVALID_PROGRAM";
    case PI_ERROR_INVALID_PROGRAM_EXECUTABLE:
      return "PI_ERROR_INVALID_PROGRAM_EXECUTABLE";
    case PI_ERROR_INVALID_KERNEL_NAME:
      return "PI_ERROR_INVALID_KERNEL_NAME";
    case PI_ERROR_INVALID_KERNEL:
      return "PI_ERROR_INVALID_KERNEL";
    case PI_ERROR_INVALID_ARG_VALUE:
      return "PI_ERROR_INVALID_ARG_VALUE";
    case PI_ERROR_INVALID_KERNEL_ARGS:
      return "PI_ERROR_INVALID_KERNEL_ARGS";
    case PI_ERROR_INVALID_WORK_DIMENSION:
      return "PI_ERROR_INVALID_WORK_DIMENSION";
    case PI_ERROR_INVALID_WORK_GROUP_SIZE:
      return "PI_ERROR_INVALID_WORK_GROUP_SIZE";
    case PI_ERROR_INVALID_WORK_ITEM_SIZE:
      return "PI_ERROR_INVALID_WORK_ITEM_SIZE";
    case PI_ERROR_INVALID_EVENT_WAIT_LIST:
      return "PI_ERROR_INVALID_EVENT_WAIT_LIST";
    case PI_ERROR_INVALID_EVENT:
      return "PI_ERROR_INVALID_EVENT";
    case PI_ERROR_INVALID_OPERATION:
      return "PI_ERROR_INVALID_OPERATION";
    case PI_ERROR_INVALID_BUFFER_SIZE:
      return "PI_ERROR_INVALID_BUFFER_SIZE";
    case PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE:
      return  "Command failed to enqueue/execute";
    case PI_ERROR_COMMAND_EXECUTION_FAILURE:
      return  "The plugin has emitted a backend specific error";      
    case PI_ERROR_PLUGIN_SPECIFIC_ERROR:
      return  "Function exists but address is not available";    
    default :
      return "Unknown PI error";
  }      
}

inline pi_image_channel_type DTypeToSYCLChannelType(DLDataType data_type) {
  DataType dtype(data_type);
  if (dtype == DataType::Float(32)) {
    return PI_IMAGE_CHANNEL_TYPE_FLOAT ;
  } else if (dtype == DataType::Float(16)) {
    return PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
  } else if (dtype == DataType::Int(8)) {
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
  } else if (dtype == DataType::Int(16)) {
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
  } else if (dtype == DataType::Int(32)) {
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
  } else if (dtype == DataType::UInt(8)) {
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  } else if (dtype == DataType::UInt(16)) {
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
  } else if (dtype == DataType::UInt(32)) {
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
  }
  LOG(FATAL) << "data type is not supported in SYCL runtime yet: " << dtype;
  return PI_IMAGE_CHANNEL_TYPE_FLOAT;
}


/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CHECK_ERROR(e)   \
  {  ICHECK(e == PI_SUCCESS) << "SYCL Error, code=" << e << ": " << syclT::SYCLGetErrorString(e);  }

#define SYCL_CALL(func)  \
{                        \
  pi_result e = (func);     \
  SYCL_CHECK_ERROR(e); \
}

class SYCLThreadEntry;

/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace : public DeviceAPI {
 public:
  // type key
  std::string type_key;
  // global platform
  // pi_platform;
  sycl::platform platform;
  // global platform name
  std::string platform_name;
  // global context of this process
  // pi_context;
  sycl::context context;
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  // std::vector<pi_device> devices;
  std::vector<sycl::device> devices;
  // the queues
  // std::vector<pi_queue> queues;
  std::vector<sycl::queue> queues;
  // the events
  // std::vector<std::vector<pi_event>> events;
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
    
    // if (context != nullptr) {
    //   SYCL_CALL(piContextRelease(context));
    // }
  }
  // Initialzie the device.
  void Init(const std::string& type_key, const std::string& device_type,
            const std::string& platform_name = "");
  virtual void Init() { Init("sycl", "gpu"); }
  // Check whether the context is SYCL or not.
  virtual bool IsSYCLDevice(Device dev) { return dev.device_type == kDLSYCL; }
  // get the queue of the device
  sycl::queue GetQueue(Device dev) {
    ICHECK(IsSYCLDevice(dev));
    this->Init();
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid SYCL device_id=" << dev.device_id;
    return queues[dev.device_id];
  }
  // get the event queue of the context
  std::vector<sycl::event>& GetEventQueue(Device dev) {
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
  std::vector<int> syclGetDeviceIDs(std::string device_type);
};

/*! \brief Thread local workspace */
class SYCLThreadEntry {
 public:
  // The kernel entry and version.
  struct KTEntry {
    // not support
    pi_kernel kernel{nullptr};
    // std::string so_path;
    // void * so_handler = nullptr;
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
}  // namespace cl

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
  std::vector<pi_kernel> kernels_;
  // parsed kernel data, Mapping from primitive name to kernel code.
  std::unordered_map<std::string, std::string> parsed_kernels_;
  // share library handler
  void * so_handler_ = nullptr;
};

/*! \brief SYCL timer node */
/*
class SYCLTimerNode : public TimerNode {
 public:
  // Timer start
  virtual void Start() {
    syclT::SYCLWorkspace::Global()->GetEventQueue(dev_).clear();
    this->duration = 0;
    // Very first call of Start() leads to the recreation of
    // SYCL command queue in profiling mode. This allows to run profile after inference.
    recreateCommandQueue();
  }
  // Timer stop
  virtual void Stop() {
    std::vector<cl_event> evt_queue = syclT::SYCLWorkspace::Global()->GetEventQueue(dev_);
    cl_ulong start, end;
    if (syclT::SYCLWorkspace::Global()->GetEventQueue(dev_).size() > 0) {
      SYCL_CALL(clWaitForEvents(1, &(syclT::SYCLWorkspace::Global()->GetEventQueue(dev_).back())));
      for (auto& kevt : evt_queue) {
        SYCL_CALL(clGetEventProfilingInfo(kevt, PI_ERROR_PROFILING_COMMAND_START, sizeof(cl_ulong),
                                            &start, nullptr));
        SYCL_CALL(clGetEventProfilingInfo(kevt, PI_ERROR_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,
                                            nullptr));
        this->duration += (end - start);
      }
    }
  }
  virtual int64_t SyncAndGetElapsedNanos() { return this->duration; }
  // destructor
  virtual ~SYCLTimerNode() {
    // Profiling session ends, recreate clCommandQueue in non-profiling mode
    // This will disable collection of cl_events in case of executing inference after profile
    recreateCommandQueue();
  }
  // constructor
  SYCLTimerNode() {}
  explicit SYCLTimerNode(Device dev) : dev_(dev) {}

  static constexpr const char* _type_key = "SYCLTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SYCLTimerNode, TimerNode);

 private:
  int64_t duration;
  Device dev_;

  void recreateCommandQueue() {
    LOG(WARNING) << "todo, not support now";
    
    cl_command_queue_properties prop;
    if (!syclT::SYCLWorkspace::Global()->IsProfiling(dev_)) {
      prop = PI_ERROR_QUEUE_PROFILING_ENABLE;
    } else {
      prop = 0;
    }

    auto queue = syclT::SYCLWorkspace::Global()->GetQueue(dev_);

    SYCL_CALL(clFlush(queue));
    SYCL_CALL(clFinish(queue));
    SYCL_CALL(clReleaseCommandQueue(queue));

    cl_int err_code;
    sycl::device did = syclT::SYCLWorkspace::Global()->devices[dev_.device_id];
    auto profiling_queue =
        clCreateCommandQueue(syclT::SYCLWorkspace::Global()->context, did, prop, &err_code);
    OPENPI_ERROR_CHECK_ERROR(err_code);
    syclT::SYCLWorkspace::Global()->queues[dev_.device_id] = profiling_queue;
    
  }
};
*/
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SYPI_ERROR_SYPI_ERROR_COMMON_H_
