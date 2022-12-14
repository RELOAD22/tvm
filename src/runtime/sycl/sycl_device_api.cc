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
#include <dmlc/thread_local.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include "sycl_common.h"

namespace tvm {
namespace runtime {
namespace syclT {

std::string syclGetPlatformInfo(cl_platform_id pid, cl_platform_info param_name);
std::string syclGetDeviceInfo(cl_device_id pid, cl_device_info param_name);

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
ImageInfo syclGetImageInfo(const syclT::syclBufferDescriptor* desc, const DLTensor* tensor) {
  ImageInfo info{};
  LOG(WARNING) << "todo, not support now";
  return info;
}

syclT::syclBufferDescriptor::MemoryLayout syclT::syclBufferDescriptor::MemoryLayoutFromScope(
    Optional<String> mem_scope) {
  LOG(WARNING) << "todo, not support now";
  return syclT::syclBufferDescriptor::MemoryLayout::kBuffer1D;
}

String syclT::syclBufferDescriptor::ScopeFromMemoryLayout(syclT::syclBufferDescriptor::MemoryLayout layout) {
  LOG(WARNING) << "todo, not support now";
  return "";
}

SYCLThreadEntry* SYCLWorkspace::GetThreadEntry() { return syclT::SYCLThreadEntry::ThreadLocal(); }

SYCLWorkspace* SYCLWorkspace::Global() {
  static SYCLWorkspace* inst = new SYCLWorkspace();
  return inst;
}

void SYCLWorkspace::SetDevice(Device dev) { GetThreadEntry()->device.device_id = dev.device_id; }

void SYCLWorkspace::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind != kExist) {
    ICHECK(index < devices.size()) << "Invalid device id " << index;
  }
  switch (kind) {
    case kExist: {
      *rv = static_cast<int>(index < devices.size());
      break;
    }
    case kMaxThreadsPerBlock: {
      size_t max_work_group_size = this->devices[index].get_info<sycl::info::device::max_work_group_size>();
      *rv = static_cast<int64_t>(max_work_group_size);
      break;
    }
    case kWarpSize: {
      // warp_size in cuda is equivalent to max_subgroup_size in sycl
      std::vector<size_t> sub_group_sizes = this->devices[index].get_info<sycl::info::device::sub_group_sizes>();
      size_t max_sub_group_size = *max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));
      *rv = static_cast<int64_t>(max_sub_group_size);
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value = this->devices[index].get_info<sycl::info::device::local_mem_size>();
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion: {
      std::string backend_version = this->devices[index].get_info<sycl::info::device::backend_version>();
      *rv = backend_version;
      break;
    }
    case kDeviceName: {
      std::string device_name = this->devices[index].get_info<sycl::info::device::name>();
      *rv = device_name;
      break;
    }
    case kMaxClockRate: {
      cl_uint max_clock_frequency = this->devices[index].get_info<sycl::info::device::max_clock_frequency>();
      // SYCL returns the clock rate in MHz, while CUDA/ROCm return the
      // clock rate in kHz.  Converting to the same units for each.
      *rv = static_cast<int32_t>(max_clock_frequency * 1000);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value = this->devices[index].get_info<sycl::info::device::max_compute_units>();
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      sycl::id<3> max_work_item_sizes = this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << max_work_item_sizes[0] << ", " << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]";
      *rv = ss.str();
      break;
    }
    case kMaxRegistersPerBlock:
      return;
    case kGcnArch:
      return;
    case kApiVersion: {
      return;
    }
    case kDriverVersion: {
      std::string driver_version = this->devices[index].get_info<sycl::info::device::driver_version>();
      *rv = driver_version;
      break;
    }
    /*
    case kCacheLineSize: {
      cl_uint cache_line_bytes = this->devices[index].get_info<sycl::info::device::global_mem_cache_line_size>();
      *rv = static_cast<int64_t>(cache_line_bytes);
      break;
    }
    */
  }
}

void* SYCLWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  VLOG(1) << "allocating " << size << "bytes device memory";
  void* ret;
  SYCL_CALL(ret = sycl::malloc_device(size, this->GetQueue(dev)))
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
  /*
  syclT::syclBufferDescriptor* desc = new syclT::syclBufferDescriptor(mem_scope);
  size_t axis = DefaultTextureLayoutSeparator(ndim, mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape, ndim, axis);
  desc->buffer = AllocTexture(dev, texture.width, texture.height, dtype);
  return desc;
  */
 return nullptr;
}

void SYCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  //std::cout<<dev.device_type<<std::endl;
  if(IsSYCLDevice(dev)){
    SYCL_CALL(sycl::free(ptr, this->GetQueue(dev)))
  }else{
    LOG(WARNING) << "not sycl device:"<<dev.device_type;
  }
  
  // We have to make sure that the memory object is not in the command queue
  // for some OpenCL platforms.
  /*
  OPENCL_CALL(clFinish(this->GetQueue(dev)));

  syclT::syclBufferDescriptor* desc = static_cast<syclT::syclBufferDescriptor*>(ptr);
  OPENCL_CALL(clReleaseMemObject(desc->buffer));
  delete desc;
  */
}

cl_mem SYCLWorkspace::AllocTexture(Device dev, size_t width, size_t height,
                                     DLDataType type_hint) {
  this->Init();
  LOG(WARNING) << "todo, not support now";
  /*
  ICHECK(context != nullptr) << "No SYCL device";
  cl_int err_code;
  cl_channel_type cl_type = DTypeToOpenCLChannelType(type_hint);
  cl_image_format format = {CL_RGBA, cl_type};
  cl_image_desc descriptor = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0};
  cl_mem mptr =
      clCreateImage(this->context, CL_MEM_READ_WRITE, &format, &descriptor, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
  */
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
  char* from_data = (char*)from->data;
  size_t from_offset = from->byte_offset;
  from_data += from_offset;
  char* to_data = (char*)to->data;
  size_t to_offset = to->byte_offset;
  to_data += to_offset;
  ICHECK(from_size == to_size) << "TVMArrayCopyFromTo: The size must exactly match";
  if (IsSYCLDevice(from->device) && IsSYCLDevice(to->device)){
    this->GetQueue(to->device).memcpy(to_data, from_data, from_size).wait();
  }else if (IsSYCLDevice(from->device) && to->device.device_type == kDLCPU){
    this->GetQueue(from->device).memcpy(to_data, from_data, from_size).wait();
  }else if (from->device.device_type == kDLCPU && IsSYCLDevice(to->device)){
    this->GetQueue(to->device).memcpy(to_data, from_data, from_size).wait();
  }else {
    LOG(FATAL) << "Expect copy from/to SYCL or between SYCL";
  }

  /*
  if (IsSYCLDevice(from->device) && IsSYCLDevice(to->device)) {
    const auto* from_desc = static_cast<const syclT::syclBufferDescriptor*>(from->data);
    ICHECK(from_desc->layout == syclT::syclBufferDescriptor::MemoryLayout::kBuffer1D)
        << "Device to device copying is currently only implemented for SYCL buffer storage";
    auto* to_desc = static_cast<syclT::syclBufferDescriptor*>(to->data);
    OPENCL_CALL(clEnqueueCopyBuffer(this->GetQueue(to->device), from_desc->buffer, to_desc->buffer,
                                    from->byte_offset, to->byte_offset, nbytes, 0, nullptr,
                                    nullptr));
  } else if (IsSYCLDevice(from->device) && to->device.device_type == kDLCPU) {
    const auto* from_desc = static_cast<const syclT::syclBufferDescriptor*>(from->data);
    switch (from_desc->layout) {
      case syclT::syclBufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueReadBuffer(
            this->GetQueue(from->device), from_desc->buffer, CL_FALSE, from->byte_offset, nbytes,
            static_cast<char*>(to->data) + to->byte_offset, 0, nullptr, nullptr));
        break;
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DActivation:
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DWeight:
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DNHWC:
        auto image_info = syclGetImageInfo(from_desc, from);
        // TODO(csullivan): Support calculating row_pitch correctly in the case of reuse.
        // Note that when utilizing texture pools for memory reuse, the allocated image
        // size can be larger than the size to be read.
        OPENCL_CALL(clEnqueueReadImage(
            this->GetQueue(from->device), from_desc->buffer, CL_FALSE, image_info.origin,
            image_info.region, image_info.row_pitch, image_info.slice_pitch,
            static_cast<char*>(to->data) + to->byte_offset, 0, nullptr, nullptr));
        break;
    }
    OPENCL_CALL(clFinish(this->GetQueue(from->device)));
  } else if (from->device.device_type == kDLCPU && IsSYCLDevice(to->device)) {
    auto* to_desc = static_cast<syclT::syclBufferDescriptor*>(to->data);
    switch (to_desc->layout) {
      case syclT::syclBufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueWriteBuffer(
            this->GetQueue(to->device), to_desc->buffer, CL_FALSE, to->byte_offset, nbytes,
            static_cast<const char*>(from->data) + from->byte_offset, 0, nullptr, nullptr));
        break;
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DActivation:
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DWeight:
      case syclT::syclBufferDescriptor::MemoryLayout::kImage2DNHWC:
        auto image_info = syclGetImageInfo(to_desc, to);
        OPENCL_CALL(clEnqueueWriteImage(
            this->GetQueue(to->device), to_desc->buffer, CL_FALSE, image_info.origin,
            image_info.region, image_info.row_pitch, image_info.slice_pitch,
            static_cast<const char*>(from->data) + from->byte_offset, 0, nullptr, nullptr));
        break;
    }
    OPENCL_CALL(clFinish(this->GetQueue(to->device)));
  } else {
    LOG(FATAL) << "Expect copy from/to SYCL or between SYCL";
  }*/
}

void SYCLWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  /*
  ICHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(dev)));*/
}

void* SYCLWorkspace::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(dev, size);
}

void SYCLWorkspace::FreeWorkspace(Device dev, void* data) {
  GetThreadEntry()->pool.FreeWorkspace(dev, data);
}

typedef dmlc::ThreadLocalStore<SYCLThreadEntry> SYCLThreadStore;

SYCLThreadEntry* SYCLThreadEntry::ThreadLocal() { return SYCLThreadStore::Get(); }

std::string syclGetPlatformInfo(cl_platform_id pid, cl_platform_info param_name) {
  LOG(WARNING) << "todo, not support now";
  return "";
}

std::string syclGetDeviceInfo(cl_device_id pid, cl_device_info param_name) {
  LOG(WARNING) << "todo, not support now";
  return "";
}


std::vector<int> SYCLWorkspace::syclGetDeviceIDs(std::string device_type) {
  sycl::info::device_type dtype = sycl::info::device_type::all;
  if (device_type == "cpu") dtype = sycl::info::device_type::cpu;
  if (device_type == "gpu") dtype = sycl::info::device_type::gpu;
  if (device_type == "accelerator") dtype = sycl::info::device_type::accelerator;
  std::vector<int> device_ids;
  for(int id=0; id<this->devices.size(); id++){
    if(this->devices[id].get_info<sycl::info::device::device_type>() == dtype){
      device_ids.push_back(id);
    }
  }
  return device_ids;
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
  if(platforms.size()==0){
    LOG(WARNING) << "No SYCL platform matched given existing options ...";
    return;
  }
  for (auto &platform : platforms) {
    std::string name = platform.get_info<sycl::info::platform::name>();
    if (name.find(platform_name) == std::string::npos) {
      continue;
    }
    if(name.find("CUDA") != std::string::npos){
      std::vector<sycl::device> devices;
      if(device_type=="gpu"){
        devices = platform.get_devices(sycl::info::device_type::gpu);
      }else{
        LOG(WARNING) << "not support device";
      }
      if (devices.size() > 0){
        this->platform = platform;
        this->platform_name = name;
        this->devices = devices;
        this->device_type = device_type;
        have_platform = true;
        break;
      }
    }
  }
  if (!have_platform) {
    LOG(WARNING) << "No CUDA device";
    return;
  }
  //create context、queues
  this->context = sycl::context(this->platform);

  auto exception_handler = [](sycl::exception_list exceptions) {
    for (const std::exception_ptr &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception &e) {
        LOG(FATAL) << "Caught asynchronous SYCL exception:" << e.what();
      }
    }
  };
  for (size_t i = 0; i < this->devices.size(); ++i) {
    this->queues.push_back(sycl::queue(this->devices[i], exception_handler));
  }
  this->events.resize(this->devices.size());
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

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
