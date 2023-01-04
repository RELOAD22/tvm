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
 * \file sycl_device_api.cc
 */
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>
#include <CL/sycl.hpp>

#include "sycl_common.h"

namespace tvm {
namespace runtime {
namespace syclT {

std::string syclGetPlatformInfo(cl_platform_id pid, pi_platform_info param_name);
std::string syclGetDeviceInfo(cl_platform_id pid, pi_platform_info param_name);

struct ImageInfo {
  size_t origin[3] = {};
  size_t region[3] = {};
  size_t row_pitch = 0;
  size_t slice_pitch = 0;
};

/*!
 * \brief Utility to apply a memory layout specific lowering convention
 * to infer the physical shape from the provided DLTensor's logical shape.
 * \param desc Descriptor which contains the buffer and layout tag.
 * \param The DLTensor used to infer the tensors physical shape.
 */
ImageInfo syclGetImageInfo(const syclBufferDescriptor* desc, const DLTensor* tensor) {
  ImageInfo info{};
  LOG(WARNING) << "todo, not support now";
  return info;
}

syclBufferDescriptor::MemoryLayout syclBufferDescriptor::MemoryLayoutFromScope(
    Optional<String> mem_scope) {
  LOG(WARNING) << "todo, not support now";
  return syclBufferDescriptor::MemoryLayout::kBuffer1D;
}

String syclBufferDescriptor::ScopeFromMemoryLayout(syclBufferDescriptor::MemoryLayout layout) {
  LOG(WARNING) << "todo, not support now";
  return "";
}

SYCLThreadEntry* SYCLWorkspace::GetThreadEntry() { return SYCLThreadEntry::ThreadLocal(); }

SYCLWorkspace* SYCLWorkspace::Global() {
  static SYCLWorkspace* inst = new SYCLWorkspace();
  return inst;
}

void SYCLWorkspace::SetDevice(Device dev) { 
  VLOG(1) << "Device id : " << dev.device_id << std::endl;
  GetThreadEntry()->device.device_id = dev.device_id; 
}

void SYCLWorkspace::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < devices.size());
    return;
  }
  ICHECK_LT(index, devices.size()) << "Invalid device id " << index;
  switch (kind) {
    case kExist:
      break;
    case kMaxThreadsPerBlock: {
      sycl::id<1> id_value = this->devices[index].get_info<sycl::info::device::max_work_item_sizes<1>>();
      int value = id_value.get(1);
      *rv = static_cast<int32_t>(value);
      return;
    }
    case kWarpSize:{
      const int warp_size = dmlc::GetEnv("TVM_SYCL_WARP_SIZE", 1);
      *rv = static_cast<int32_t>(warp_size);
      break;
    }
    case kMaxSharedMemoryPerBlock:{
      sycl::id<1> id_value = this->devices[index].get_info<sycl::info::device::local_mem_size>();
      int value = id_value.get(1);
      *rv = static_cast<int32_t>(value);
      return;
    }
    case kComputeVersion:{
      std::string value = this->devices[index].get_info<sycl::info::device::backend_version>();
      *rv = std::string(value);
      return ;
    }
    case kDeviceName:{
      std::string value = this->devices[index].get_info<sycl::info::device::name>();
      *rv = std::string(value);
      return ;
    }
    case kMaxClockRate:{
      uint32_t value = this->devices[index].get_info<sycl::info::device::max_clock_frequency>();
      *rv = static_cast<int32_t>(value);
      return ;
    }
    case kMultiProcessorCount:{
      size_t value = this->devices[index].get_info<sycl::info::device::max_work_group_size>();
      *rv = static_cast<int32_t>(value);
      return ;
    }
    case kMaxThreadDimensions:{
      *rv = static_cast<int32_t>(3);
      return ;
    }
    case kMaxRegistersPerBlock:{
      return ;
    }
    case kGcnArch:{
      return ;
    }
    case kApiVersion:{
      return ;
    }
    case kDriverVersion:{
      auto value = this->devices[index].get_info<sycl::info::device::driver_version>();
      *rv = std::string(value);
      return ;
    }
  }
}


void* SYCLWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  VLOG(1) << "sycl device allocating " << size << " bytes share memory";
  VLOG(1) << "alloc sycl device id is " << dev.device_id << std::endl;
  VLOG(1) << "alloc sycl device type is " << dev.device_type << std::endl;
  VLOG(1) << "alloc sycl device alignment is " << alignment << std::endl;
  // void* ret = sycl::malloc_shared(size, this->devices[dev.device_id], this->context);
  void* ret = nullptr;
  if(dev.device_type == kDLCPU ){
    ret = sycl::aligned_alloc_host(alignment,size,this->contexts[dev.device_id]);
  }else if(dev.device_type == kDLSYCL){
    ret = sycl::aligned_alloc_device(alignment,size,this->devices[dev.device_id],this->contexts[dev.device_id]);
  }else{
    std::cerr<<"unknown device type : "<<dev.device_type<<std::endl;
  }
  // void* ret = sycl::aligned_alloc_shared(alignment,size,this->devices[dev.device_id],this->context);
  if(ret == nullptr)
    LOG(ERROR) << "allgn alloc memory failure!"<<std::endl;
  VLOG(1) << "alloc sycl device pointer address is " << ret << std::endl;
  return ret;
}

void* SYCLWorkspace::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "global") {
    return DeviceAPI::AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
  }
  ICHECK(IsTextureStorage(std::string(mem_scope.value())))
      << "Device does not support allocate data space with "
      << "specified memory scope: " << mem_scope.value();

  ICHECK(ndim > 2) << "Shape for texture allocation must be at least rank 3; "
                   << "provided shape is rank " << ndim;

  LOG(WARNING) << "todo, not support now";
 return nullptr;
}

void SYCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  SYCL_CALL(this->GetQueue(dev).wait_and_throw());
  if(!IsSYCLDevice(dev)){
    VLOG(1) << "free not sycl device : "<<dev.device_type;
    LOG(WARNING) << "free not sycl device:"<<dev.device_type;
    return ;
  }else{
    //IsSYCLDevice(dev) == true
    VLOG(1) << "free sycl device id is " << dev.device_id << std::endl;
    VLOG(1) << "free sycl device type is " << dev.device_type << std::endl;
    VLOG(1) << "free sycl device pointer address is " << ptr << std::endl;
  }
  sycl::queue queue = this->GetQueue(dev);
  sycl::free(ptr, queue);
}

cl_mem SYCLWorkspace::AllocTexture(Device dev, size_t width, size_t height,
                                     DLDataType type_hint) {
  this->Init();
  LOG(WARNING) << "todo, not support now";
  cl_mem mptr;
  return mptr;
}

void* SYCLWorkspace::AllocTextureWorkspace(Device dev, size_t width, size_t height,
                                             DLDataType type_hint) {
  return GetThreadEntry()->texture_pool.AllocTexture(dev, width, height, type_hint);
}

void SYCLWorkspace::FreeTextureWorkspace(Device dev, void* ptr) {
  GetThreadEntry()->texture_pool.FreeTexture(dev, ptr);
}


void SYCLWorkspace::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  ICHECK_EQ(from_size, to_size);
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";

  size_t from_offset = from->byte_offset;
  size_t to_offset = to->byte_offset;
  VLOG(1) << "from device " << from->device.device_id << " type : "<< from->device.device_type<<std::endl;
  VLOG(1) << "to device " << to->device.device_id << " type : "<< to->device.device_type<<std::endl;

  VLOG(1) << "before convert from device data pointer address : " << from->data << std::endl;
  VLOG(1) << "before convert to device data pointer address : " << to->data << std::endl;
  
  from->data = static_cast<char*>(from->data) + from->byte_offset;
  to->data = static_cast<char*>(to->data) + to->byte_offset;

  ICHECK(from_size == to_size) << "TVMArrayCopyFromTo: The size must exactly match";

  VLOG(1) << "after convert from device data pointer address : " << from->data << std::endl;
  VLOG(1) << "after convert to device data pointer address : " << to->data << std::endl;
  if (IsSYCLDevice(from->device) && IsSYCLDevice(to->device)){
    auto queue = this->GetQueue(to->device);
    auto event = queue.memcpy(to->data,from->data,from_size);
    SYCL_CALL(event.wait());
  }else if (IsSYCLDevice(from->device) && to->device.device_type == kDLCPU){
    auto queue = this->GetQueue(from->device);
    auto event = queue.memcpy(to->data,from->data,from_size);
    SYCL_CALL(event.wait());
    // SYCL_CALL(this->GetQueue(from->device).memcpy(to->data, from->data, from_size).wait());
  }else if (from->device.device_type == kDLCPU && IsSYCLDevice(to->device)){
    auto queue = this->GetQueue(to->device);
    auto event = queue.memcpy(to->data,from->data,from_size);
    SYCL_CALL(event.wait());
    // SYCL_CALL(this->GetQueue(to->device).memcpy(to->data, from->data, from_size).wait());
  }else {
    LOG(FATAL) << "Expect copy from/to SYCL or between SYCL";
  }
}


void SYCLWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  ICHECK(stream == nullptr);
  SYCL_CALL(this->GetQueue(dev).wait_and_throw());
}

void* SYCLWorkspace::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(dev, size);
}

void SYCLWorkspace::FreeWorkspace(Device dev, void* data) {
  GetThreadEntry()->pool.FreeWorkspace(dev, data);
}

typedef dmlc::ThreadLocalStore<SYCLThreadEntry> SYCLThreadStore;

SYCLThreadEntry* SYCLThreadEntry::ThreadLocal() { return SYCLThreadStore::Get(); }

std::string syclGetPlatformInfo(cl_platform_id pid, pi_platform_info param_name) {
  LOG(WARNING) << "todo, not support now";
  return "";
}

std::string syclGetDeviceInfo(cl_device_id pid, pi_device_info param_name) {
  LOG(WARNING) << "todo, not support now";
  return "";
}


void SYCLWorkspace::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {                     
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  this->type_key = type_key;

  // sycl add
  // look for matched platform
  bool have_platform = false;
  auto platforms = sycl::platform::get_platforms();
  if(platforms.size() <= 1){
    LOG(ERROR) << "No device SYCL platform matched given existing options ...";
    return;
  }
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (const std::exception_ptr &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };
  for (auto &platform : platforms) {
    if(device_type == "gpu"){
        std::string platform_name = platform.get_info<sycl::info::platform::name>();
        // neither NVIDIA CUDA BACKEND nor AMD HIP BACKEND
        if(platform_name.find("BACKEND") == std::string::npos)
          continue;
        std::vector<sycl::device> devices = platform.get_devices(sycl::info::device_type::gpu);
        if(devices.size() > 0){
          if(devices.size() > 1)
            LOG(WARNING) << "No Support Sub Devices";
          this->platforms.push_back(platform);
          this->devices.insert(this->devices.end(),devices.begin(),devices.end());
          this->platform_names.push_back(platform_name);
          this->device_type = device_type;
          sycl::device dev = devices[0];
          sycl::context ctx = sycl::context(dev,exception_handler);
          this->contexts.push_back(ctx);
          sycl::queue queue = sycl::queue(ctx,devices[0]);
          this->queues.push_back(queue);
          have_platform = true;
        }
    }
  }
  this->events.resize(this->devices.size());
  VLOG(1) << "platforms size : " << this->platforms.size() << std::endl;
  VLOG(1) << "devices size : " << this->devices.size() << std::endl;
  VLOG(1) << "contexts size : " << this->contexts.size() << std::endl;
  VLOG(1) << "queues size : " << this->queues.size() << std::endl;
  initialized_ = true;
}

TVM_REGISTER_GLOBAL("device_api.sycl.alloc_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  int32_t dtype_code_hint = args[2];
  int32_t dtype_bits_hint = args[3];
  std::string scope = args[4];
  CHECK(scope.find("texture") != std::string::npos);
  int64_t ndim = args[5];
  CHECK_EQ(ndim, 2);
  int64_t* shape = static_cast<int64_t*>(static_cast<void*>(args[6]));
  int64_t width = shape[0];
  int64_t height = shape[1];

  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  SYCLWorkspace* ptr = SYCLWorkspace::Global();
  *rv = ptr->AllocTextureWorkspace(dev, static_cast<size_t>(width), static_cast<size_t>(height),
                                   type_hint);
});

TVM_REGISTER_GLOBAL("device_api.sycl.free_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  std::string scope = args[2];
  CHECK(scope.find("texture") != std::string::npos);
  void* data = args[3];
  SYCLWorkspace* ptr = SYCLWorkspace::Global();
  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  ptr->FreeTextureWorkspace(dev, data);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.sycl").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = SYCLWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

#ifdef USE_PROFILER
TVM_REGISTER_OBJECT_TYPE(SYCLTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.sycl").set_body_typed([](Device dev) {
  return Timer(make_object<SYCLTimerNode>(dev));
});
#endif

}  // namespace sycl
}  // namespace runtime
}  // namespace tvm
