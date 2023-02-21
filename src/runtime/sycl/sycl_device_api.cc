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
#include <tvm/runtime/registry.h>

#include "sycl_common.h"

namespace tvm {
namespace runtime {
namespace syclT {

SYCLThreadEntry* SYCLWorkspace::GetThreadEntry() { return syclT::SYCLThreadEntry::ThreadLocal(); }

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
  void* ret = nullptr;
  if(IsSYCLDevice(dev)){
    this->GetQueue(dev).wait_and_throw();
    VLOG(1) << "allocating " << size << "bytes device memory, ";
    SYCL_CALL(ret = sycl::malloc_device(size, this->GetQueue(dev)))
  }else{
    LOG(ERROR) << "allocate in not sycl device:"<<dev.device_type;
  }
  if(ret == nullptr)
    LOG(ERROR) << "allocate sycl device memory failure!"<<std::endl;
  VLOG(1) << "allocate sycl device memory from "<< ret << std::endl;
  return ret;
}

void SYCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  if(IsSYCLDevice(dev)){
    this->GetQueue(dev).wait_and_throw();
    VLOG(1) << "free sycl device memory from " << ptr << std::endl;
    SYCL_CALL(sycl::free(ptr, this->GetQueue(dev)))
  }else{
    LOG(ERROR) << "free in not sycl device:"<<dev.device_type;
  }
}

void SYCLWorkspace::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  ICHECK(from_size == to_size) << "TVMArrayCopyFromTo: The size must exactly match";
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";
  char* from_data = (char*)from->data;
  size_t from_offset = from->byte_offset;
  from_data += from_offset;
  char* to_data = (char*)to->data;
  size_t to_offset = to->byte_offset;
  to_data += to_offset;

  VLOG(1) << "from device " << from->device.device_id << " type : "<< from->device.device_type<<std::endl;
  VLOG(1) << "to device " << to->device.device_id << " type : "<< to->device.device_type<<std::endl;
  VLOG(1) << "before convert from device data pointer address : " << from_data << std::endl;
  VLOG(1) << "before convert to device data pointer address : " << to_data << std::endl;
  VLOG(1) << "after convert from device data pointer address : " << from_data+from_size << std::endl;
  VLOG(1) << "after convert to device data pointer address : " << to_data+from_size << std::endl;
  
  if (IsSYCLDevice(from->device) && IsSYCLDevice(to->device)){
    this->GetQueue(to->device).memcpy(to_data, from_data, from_size).wait();
  }else if (IsSYCLDevice(from->device) && to->device.device_type == kDLCPU){
    this->GetQueue(from->device).memcpy(to_data, from_data, from_size).wait();
  }else if (from->device.device_type == kDLCPU && IsSYCLDevice(to->device)){
    this->GetQueue(to->device).memcpy(to_data, from_data, from_size).wait();
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



void SYCLWorkspace::Init(const std::string& device_type, const std::string& platform_name) {                     
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);

  // sycl add
  // look for matched platform
  bool have_platform = false;
  auto platforms = sycl::platform::get_platforms();
  std::string target_platform = SYCL_GPU_TYPE; //from config.cmake, possible values: "nvidia", "amd".
  std::string key;
  if(target_platform=="nvidia"){
    key = "CUDA";
  }else if(target_platform=="amd"){
    key = "HIP"; // todo
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
      std::string name = platform.get_info<sycl::info::platform::name>();
      if (name.find(platform_name) == std::string::npos)
        continue;
      // neither NVIDIA CUDA BACKEND nor AMD HIP BACKEND
      if(name.find(key) == std::string::npos)
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
        //create context and queue
        sycl::context ctx = sycl::context(dev,exception_handler);
        this->contexts.push_back(ctx);
        sycl::queue queue = sycl::queue(ctx,devices[0]);
        this->queues.push_back(queue);
        have_platform = true;          
      }
    }
  }
  if (!have_platform) {
    LOG(ERROR) << "No valid gpu device/platform matched given existing options ...";
    return;
  }
  
  this->events.resize(this->devices.size());
  initialized_ = true;
  VLOG(1) << "devices size : " << this->devices.size() << std::endl;
}

sycl::queue SYCLWorkspace::GetQueue(Device dev) {
  ICHECK(IsSYCLDevice(dev));
  this->Init();
  ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size()) 
      << "Invalid SYCL device_id=" << dev.device_id;
  return queues[dev.device_id];
}

std::vector<sycl::event>& SYCLWorkspace::GetEventQueue(Device dev) {
  ICHECK(IsSYCLDevice(dev));
  this->Init();
  ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
      << "Invalid SYCL device_id=" << dev.device_id;
  return events[dev.device_id];
}

bool SYCLWorkspace::IsProfiling(Device dev) {
  sycl::queue queue = GetQueue(dev);
  return queue.has_property<sycl::property::queue::enable_profiling>();
}

TVM_REGISTER_GLOBAL("device_api.sycl").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = SYCLWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_OBJECT_TYPE(SYCLTimerNode);
TVM_REGISTER_GLOBAL("profiling.timer.sycl").set_body_typed([](Device dev) {
  return Timer(make_object<SYCLTimerNode>());
});

}  // namespace SYCL
}  // namespace runtime
}  // namespace tvm
