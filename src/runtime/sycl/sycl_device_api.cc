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

#include "sycl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

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
ImageInfo syclGetImageInfo(const cl::syclBufferDescriptor* desc, const DLTensor* tensor) {
  ImageInfo info{};
  ICHECK(tensor->dtype.lanes == 1) << "Image dtype has lanes: " << tensor->dtype.lanes;

  info.origin[0] = info.origin[1] = info.origin[2] = 0;
  info.row_pitch = 0;
  info.slice_pitch = 0;

  size_t axis = DefaultTextureLayoutSeparator(
      tensor->ndim, cl::syclBufferDescriptor::ScopeFromMemoryLayout(desc->layout));
  auto texture_shape = ApplyTexture2DFlattening<int64_t>(tensor->shape, tensor->ndim, axis);
  info.region[0] = texture_shape.width;
  info.region[1] = texture_shape.height;
  info.region[2] = 1;
  return info;
}

cl::syclBufferDescriptor::MemoryLayout cl::syclBufferDescriptor::MemoryLayoutFromScope(
    Optional<String> mem_scope) {
  if (!mem_scope.defined()) {
    return cl::syclBufferDescriptor::MemoryLayout::kBuffer1D;
  } else if (mem_scope.value() == "global.texture") {
    return cl::syclBufferDescriptor::MemoryLayout::kImage2DActivation;
  } else if (mem_scope.value() == "global.texture-weight") {
    return cl::syclBufferDescriptor::MemoryLayout::kImage2DWeight;
  } else if (mem_scope.value() == "global.texture-nhwc") {
    return cl::syclBufferDescriptor::MemoryLayout::kImage2DNHWC;
  }
  LOG(FATAL) << "No memory layout defined for memory of scope: " << mem_scope.value();
  return cl::syclBufferDescriptor::MemoryLayout::kBuffer1D;
}

String cl::syclBufferDescriptor::ScopeFromMemoryLayout(cl::syclBufferDescriptor::MemoryLayout layout) {
  switch (layout) {
    case cl::syclBufferDescriptor::MemoryLayout::kBuffer1D:
      return "global";
    case cl::syclBufferDescriptor::MemoryLayout::kImage2DActivation:
      return "global.texture";
    case cl::syclBufferDescriptor::MemoryLayout::kImage2DWeight:
      return "global.texture-weight";
    case cl::syclBufferDescriptor::MemoryLayout::kImage2DNHWC:
      return "global.texture-nhwc";
  }
  LOG(FATAL) << "No scope corresponding to the provided memory layout: "
             << static_cast<int>(layout);
  return "";
}

SYCLThreadEntry* SYCLWorkspace::GetThreadEntry() { return SYCLThreadEntry::ThreadLocal(); }

SYCLWorkspace* SYCLWorkspace::Global() {
  static SYCLWorkspace* inst = new SYCLWorkspace();
  return inst;
}

void SYCLWorkspace::SetDevice(Device dev) { GetThreadEntry()->device.device_id = dev.device_id; }

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
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                  &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      /* TODO: the warp size of OpenCL device is not always 1
               e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
               corresponding to the number of SIMD entries the heardware configures.
               We need to figure out a way to query this information from the hardware.
      */
      const int warp_size = dmlc::GetEnv("TVM_OPENCL_WARP_SIZE", 1);
      *rv = warp_size;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
                                  &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion: {
      // String returned is "OpenCL $MAJOR.$MINOR $VENDOR_INFO".  To
      // match other implementations, we want to return "$MAJOR.$MINOR"
      std::string ret = syclGetDeviceInfo(devices[index], CL_DEVICE_VERSION);

      const size_t version_start = 7;  // Length of initial "OpenCL " prefix to skip
      const size_t version_end = ret.find(' ', version_start);
      *rv = ret.substr(version_start, version_end - version_start);
      break;
    }
      return;
    case kDeviceName:
      *rv = syclGetDeviceInfo(devices[index], CL_DEVICE_NAME);
      break;
    case kMaxClockRate: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
                                  &value, nullptr));
      // OpenCL returns the clock rate in MHz, while CUDA/ROCm return the
      // clock rate in kHz.  Converting to the same units for each.
      *rv = static_cast<int32_t>(value * 1000);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims,
                                  nullptr));

      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
      *rv = ss.str();
      break;
    }
    case kMaxRegistersPerBlock:
      return;
    case kGcnArch:
      return;
    case kApiVersion: {
      *rv = CL_TARGET_OPENCL_VERSION;
      break;
    }
    case kDriverVersion: {
      char value[128] = {0};
      OPENCL_CALL(
          clGetDeviceInfo(devices[index], CL_DRIVER_VERSION, sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
  }
}

void* SYCLWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  void* ret;

  if (dev.device_type == kDLSYCLHost) {
    VLOG(1) << "allocating " << size << "bytes on host";
    ret = sycl::malloc_host(size, this->sycl_context);
  }else{
    VLOG(1) << "allocating " << size << "bytes share memory";
    ret = sycl::malloc_shared(size, this->sycl_device, this->sycl_context);
  }
  return ret;
  /*
  cl_int err_code;
  cl::syclBufferDescriptor* desc = new cl::syclBufferDescriptor;
  // CL_INVALID_BUFFER_SIZE if size is 0.
  if (size == 0) {
    size = 1;
  }
  desc->buffer = clCreateBuffer(this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  desc->layout = cl::syclBufferDescriptor::MemoryLayout::kBuffer1D;
  OPENCL_CHECK_ERROR(err_code);
  return desc;
  */
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

  cl::syclBufferDescriptor* desc = new cl::syclBufferDescriptor(mem_scope);
  size_t axis = DefaultTextureLayoutSeparator(ndim, mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape, ndim, axis);
  desc->buffer = AllocTexture(dev, texture.width, texture.height, dtype);
  return desc;
}

void SYCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  // We have to make sure that the memory object is not in the command queue
  // for some OpenCL platforms.
  OPENCL_CALL(clFinish(this->GetQueue(dev)));

  cl::syclBufferDescriptor* desc = static_cast<cl::syclBufferDescriptor*>(ptr);
  OPENCL_CALL(clReleaseMemObject(desc->buffer));
  delete desc;
}

cl_mem SYCLWorkspace::AllocTexture(Device dev, size_t width, size_t height,
                                     DLDataType type_hint) {
  this->Init();
  ICHECK(context != nullptr) << "No SYCL device";
  cl_int err_code;
  cl_channel_type cl_type = DTypeToOpenCLChannelType(type_hint);
  cl_image_format format = {CL_RGBA, cl_type};
  cl_image_desc descriptor = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0};
  cl_mem mptr =
      clCreateImage(this->context, CL_MEM_READ_WRITE, &format, &descriptor, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
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

  ICHECK_EQ(from_size, to_size) << "TVMArrayCopyFromTo: The size must exactly match";
  this->default_queue.memcpy(to_data, from_data, from_size).wait();
  /*
  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";

  if (IsSYCLDevice(from->device) && IsSYCLDevice(to->device)) {
    const auto* from_desc = static_cast<const cl::syclBufferDescriptor*>(from->data);
    ICHECK(from_desc->layout == cl::syclBufferDescriptor::MemoryLayout::kBuffer1D)
        << "Device to device copying is currently only implemented for SYCL buffer storage";
    auto* to_desc = static_cast<cl::syclBufferDescriptor*>(to->data);
    OPENCL_CALL(clEnqueueCopyBuffer(this->GetQueue(to->device), from_desc->buffer, to_desc->buffer,
                                    from->byte_offset, to->byte_offset, nbytes, 0, nullptr,
                                    nullptr));
  } else if (IsSYCLDevice(from->device) && to->device.device_type == kDLCPU) {
    const auto* from_desc = static_cast<const cl::syclBufferDescriptor*>(from->data);
    switch (from_desc->layout) {
      case cl::syclBufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueReadBuffer(
            this->GetQueue(from->device), from_desc->buffer, CL_FALSE, from->byte_offset, nbytes,
            static_cast<char*>(to->data) + to->byte_offset, 0, nullptr, nullptr));
        break;
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DActivation:
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DWeight:
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DNHWC:
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
    auto* to_desc = static_cast<cl::syclBufferDescriptor*>(to->data);
    switch (to_desc->layout) {
      case cl::syclBufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueWriteBuffer(
            this->GetQueue(to->device), to_desc->buffer, CL_FALSE, to->byte_offset, nbytes,
            static_cast<const char*>(from->data) + from->byte_offset, 0, nullptr, nullptr));
        break;
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DActivation:
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DWeight:
      case cl::syclBufferDescriptor::MemoryLayout::kImage2DNHWC:
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
  size_t ret_size;
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::string syclGetDeviceInfo(cl_device_id pid, cl_device_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_platform_id> syclGetPlatformIDs() {
  cl_uint ret_size;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> syclGetDeviceIDs(cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size;
  cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
  std::vector<cl_device_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
  return ret;
}

bool syclMatchPlatformInfo(cl_platform_id pid, cl_platform_info param_name, std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = syclGetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

void SYCLWorkspace::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {                     
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  if (context != nullptr) return;
  this->type_key = type_key;
  // matched platforms
  std::vector<cl_platform_id> platform_ids = cl::syclGetPlatformIDs();
  if (platform_ids.size() == 0) {
    LOG(WARNING) << "No SYCL platform matched given existing options ...";
    return;
  }
  this->platform_id = nullptr;
  for (auto platform_id : platform_ids) {
    if (!syclMatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
      continue;
    }
    std::vector<cl_device_id> devices_matched = cl::syclGetDeviceIDs(platform_id, device_type);
    if ((devices_matched.size() == 0) && (device_type == "gpu")) {
      LOG(WARNING) << "Using CPU SYCL device";
      devices_matched = cl::syclGetDeviceIDs(platform_id, "cpu");
    }
    if (devices_matched.size() > 0) {
      this->platform_id = platform_id;
      this->platform_name = cl::syclGetPlatformInfo(platform_id, CL_PLATFORM_NAME);
      this->device_type = device_type;
      this->devices = devices_matched;
      break;
    }
  }
  if (this->platform_id == nullptr) {
    LOG(WARNING) << "No SYCL device";
    return;
  }
  cl_int err_code;
  this->context = clCreateContext(nullptr, this->devices.size(), &(this->devices[0]), nullptr,
                                  nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  ICHECK_EQ(this->queues.size(), 0U);
  for (size_t i = 0; i < this->devices.size(); ++i) {
    cl_device_id did = this->devices[i];
    this->queues.push_back(clCreateCommandQueue(this->context, did, 0, &err_code));
    OPENCL_CHECK_ERROR(err_code);
  }
  this->events.resize(this->devices.size());
  initialized_ = true;
  // sycl add
  if(device_type == "gpu"){
    sycl::gpu_selector device_selector;
    this->sycl_device = sycl::device(device_selector);
  }else{
    sycl::default_selector device_selector;
    this->sycl_device = sycl::device(device_selector);
  }
  this->sycl_context = sycl::context(this->sycl_device);
  this->default_queue = sycl::queue(this->sycl_device);
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
