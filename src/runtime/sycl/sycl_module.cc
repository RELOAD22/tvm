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
 * \file sycl_module.cc
 */
#include "sycl_module.h"

#include <CL/sycl.hpp>
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <dlfcn.h>

#include "../source_utils.h"
#include "sycl_common.h"

namespace tvm {
namespace runtime {

class SYCLWrappedFunc {
 public:
  // initialize the SYCL function.
  void Init(SYCLModuleNode* m, std::string func_name, \
    size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
    VLOG(0) << "Init SYCLWrappedFunc";
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    //sptr_ = sptr;
    //entry_ = entry;
    func_name_ = func_name;
    launch_param_config_.Init(num_void_args, launch_param_tags);

    VLOG(0) << m_->GetSource("sycl");
    system("mkdir /tmp/tvm_sycl");
    std::string filepath = "/tmp/tvm_sycl/"+func_name+".cpp";
    std::string sharedlibpath = "/tmp/tvm_sycl/"+func_name+".so";
    std::ofstream myfile;
    myfile.open(filepath);
    myfile << m_->GetSource("sycl");
    myfile.close();
    std::string cmd = "$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fPIC -shared " + filepath + " -o " + sharedlibpath;
    system(cmd.c_str());
    VLOG(0) << cmd;
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    VLOG(0) << "enter sycl wrapped func operator()";
    std::string sharedlibpath = "/tmp/tvm_sycl/"+func_name_+".so";
    void *dl_handler = dlopen(sharedlibpath.c_str(), RTLD_LAZY);
    if (dl_handler == NULL)
    {
        printf("ERROR:%s:dlopen\n", dlerror());
        return;
    }
    void (*kernel_func)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) = (void (*)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args))dlsym(dl_handler, func_name_.c_str());
    if (kernel_func == NULL)
    {
        printf("ERROR:%s:dlsym\n", dlerror());
        return;
    }
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    sycl::range<3> k0_dimGrid(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    sycl::range<3> k0_dimBlock(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
    
    sycl::queue Q;
    /*


    void * void_args1[] = {(void *)&shared_array, (void *)&host_array1, (void *)&host_array2,
                                        (void *)&n, (void *)&stride, (void *)&stride1, (void *)&stride2};
    
    void *dst = void_args[0], *source1 = void_args[1], *source2 = void_args[2];
    int n = *(int *)void_args[3], stride0=*(int *)void_args[4], stride1=*(int *)void_args[5], stride2=*(int *)void_args[6];
    std::cout << "n: " << n << " stride: " << stride0 << " stride1: " << stride1 << " stride2: " << stride2 << std::endl;
    */
    
    kernel_func(Q, k0_dimGrid, k0_dimBlock, void_args);
    /*
    for (int i = 0; i < n; i++) {
        // access sharedArray on host
        std::cout << ((float *)(*(int64_t *)dst))[i] << " ";
    }
    std::cout << std::endl;
    */
    /*
    free(shared_array, Q);
    free(host_array1, Q);
    free(host_array2, Q);
    */
	  dlclose(dl_handler);
    /*
    ICHECK(w_->context != nullptr) << "No SYCL device";
    cl::SYCLThreadEntry* t = w_->GetThreadEntry();
    // get the kernel from thread local kernel table.
    if (entry_.kernel_id >= t->kernel_table.size()) {
      t->kernel_table.resize(entry_.kernel_id + 1);
    }
    const auto& e = t->kernel_table[entry_.kernel_id];
    cl_kernel kernel = e.kernel;
    if (kernel == nullptr || e.version != entry_.version) {
      kernel = m_->InstallKernel(w_, t, func_name_, entry_);
    }
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      void* arg = nullptr;
      if (args.type_codes[i] == DLDataTypeCode::kDLOpaqueHandle) {
        arg = static_cast<cl::syclBufferDescriptor*>(void_args[i])->buffer;
      } else {
        arg = void_args[i];
      }
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], arg));
    }
    cl_command_queue queue = w_->GetQueue(t->device);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(launch_param_config_.work_dim());
    for (cl_uint i = 0; i < work_dim; ++i) {
      wl.work_size[i] *= wl.work_size[i + 3];
    }
    // launch kernel

    if (w_->IsProfiling(t->device)) {
      w_->GetEventQueue(t->device).resize(w_->GetEventQueue(t->device).size() + 1);
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr,
                                         &(w_->GetEventQueue(t->device).back())));
    } else {
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr, nullptr));
    }*/
  }

 private:
  // global workspace.
  cl::SYCLWorkspace* w_;
  // The module
  SYCLModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // global kernel id in the kernel table.
  SYCLModuleNode::KTRefEntry entry_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // launch parameters config
  LaunchParamConfig launch_param_config_;
};

SYCLModuleNode::~SYCLModuleNode() {
  {
    // free the kernel ids in global table.
    std::lock_guard<std::mutex> lock(workspace_->mu);
    for (auto& kv : kid_map_) {
      workspace_->free_kernel_ids.push_back(kv.second.kernel_id);
    }
  }
  // free the kernels
  for (cl_kernel k : kernels_) {
    OPENCL_CALL(clReleaseKernel(k));
  }
  // free the programs
  for (auto& kv : programs_) {
    for (auto& program : kv.second) {
      if (program) {
        OPENCL_CALL(clReleaseProgram(program));
      }
    }
  }
}

cl::SYCLWorkspace* SYCLModuleNode::GetGlobalWorkspace() {
  return cl::SYCLWorkspace::Global();
}

PackedFunc SYCLModuleNode::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  VLOG(0) << "SYCLModuleNode::GetFunction: " << name;
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  VLOG(0) << "SYCLModuleNode::find fmap_";

  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  VLOG(0) << "SYCLModuleNode::begin SYCLWrappedFunc";

  SYCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    DLDataType t = info.arg_types[i];
    ICHECK_EQ(t.lanes, 1U);
    if (t.code == kTVMOpaqueHandle) {
      // specially store pointer type size in SYCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      ICHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  VLOG(0) << "SYCLModuleNode::begin initialize the wrapped func:";
  // initialize the wrapped func.
  f.Init(this, name, info.arg_types.size(), info.launch_param_tags);
  VLOG(0) << "SYCLModuleNode::finish initialize the wrapped func:";
  return PackFuncVoidAddr(f, info.arg_types);
}

void SYCLModuleNode::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, data_);
}

void SYCLModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
}

std::string SYCLModuleNode::GetSource(const std::string& format) {
  if (format == fmt_) return data_;
  if (fmt_ == "cl") {
    return data_;
  } else {
    return source_;
  }
}

void SYCLModuleNode::Init() {
  workspace_ = GetGlobalWorkspace();
  workspace_->Init();
  // initialize the kernel id, need to lock global table.
  std::lock_guard<std::mutex> lock(workspace_->mu);
  for (const auto& kv : fmap_) {
    const std::string& key = kv.first;
    KTRefEntry e;
    if (workspace_->free_kernel_ids.size() != 0) {
      e.kernel_id = workspace_->free_kernel_ids.back();
      workspace_->free_kernel_ids.pop_back();
    } else {
      e.kernel_id = workspace_->num_registered_kernels++;
    }
    e.version = workspace_->timestamp++;
    kid_map_[key] = e;
  }

  // split into source artifacts for each kernel
  parsed_kernels_ = SplitKernels(GetSource("cl"));
  ICHECK(!parsed_kernels_.empty()) << "The SYCL module expects a kernel delimited "
                                   << "source from code generation, but no kernel "
                                   << "delimiter was found.";
  ICHECK_EQ(fmap_.size(), parsed_kernels_.size())
      << "The number of parsed kernel sources does not match the number of kernel functions";
  // zero initialize cl_program pointers for each device kernel
  for (auto& kv : parsed_kernels_) {
    programs_.insert({kv.first, std::vector<cl_program>(workspace_->devices.size(), nullptr)});
  }
}

cl_kernel SYCLModuleNode::InstallKernel(cl::SYCLWorkspace* w, cl::SYCLThreadEntry* t,
                                          const std::string& func_name, const KTRefEntry& e) {
  std::lock_guard<std::mutex> lock(build_lock_);
  int device_id = t->device.device_id;
  if (programs_[func_name][device_id] == nullptr) {
    // create program
    if (fmt_ == "cl") {
      const char* s = parsed_kernels_[func_name].c_str();
      size_t len = parsed_kernels_[func_name].length();
      cl_int err;
      programs_[func_name][device_id] = clCreateProgramWithSource(w->context, 1, &s, &len, &err);
      OPENCL_CHECK_ERROR(err);
    } else if (fmt_ == "xclbin" || fmt_ == "awsxclbin" || fmt_ == "aocx") {
      const unsigned char* s = (const unsigned char*)data_.c_str();
      size_t len = data_.length();
      cl_int err;
      cl_device_id dev = w->devices[device_id];
      programs_[func_name][device_id] =
          clCreateProgramWithBinary(w->context, 1, &dev, &len, &s, NULL, &err);
      OPENCL_CHECK_ERROR(err);
    } else {
      LOG(FATAL) << "Unknown SYCL format " << fmt_;
    }
    // build program
    cl_int err;
    cl_device_id dev = w->devices[device_id];
    err = clBuildProgram(programs_[func_name][device_id], 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t len;
      std::string log;
      clGetProgramBuildInfo(programs_[func_name][device_id], dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &len);
      log.resize(len);
      clGetProgramBuildInfo(programs_[func_name][device_id], dev, CL_PROGRAM_BUILD_LOG, len,
                            &log[0], nullptr);
      LOG(FATAL) << "SYCL build error for device=" << dev << "\n" << log;
    }
  }
  // build kernel
  cl_int err;
  cl_kernel kernel = clCreateKernel(programs_[func_name][device_id], func_name.c_str(), &err);
  OPENCL_CHECK_ERROR(err);
  t->kernel_table[e.kernel_id].kernel = kernel;
  t->kernel_table[e.kernel_id].version = e.version;
  kernels_.push_back(kernel);
  return kernel;
}

Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source);
  //n->Init();
  return Module(n);
}

// Load module from module.
Module SYCLModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SYCLModuleCreate(data, fmt, fmap, std::string());
}

Module SYCLModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SYCLModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_sycl").set_body_typed(SYCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_syclbin").set_body_typed(SYCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_sycl").set_body_typed(SYCLModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
