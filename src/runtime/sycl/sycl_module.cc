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
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <dlfcn.h>

#include "../source_utils.h"
#include "sycl_common.h"
#include <unistd.h>

namespace tvm {
namespace runtime {

class SYCLWrappedFunc {
 public:
  // initialize the SYCL function.
  void Init(SYCLModuleNode* m, std::string func_name, \
    size_t num_void_args, const std::vector<std::string>& launch_param_tags, void *so_handler) {
    VLOG(0) << "Init SYCLWrappedFunc";
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    //sptr_ = sptr;
    func_name_ = func_name;
    so_handler_ = so_handler;
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    VLOG(0) << "enter sycl wrapped func operator()";
    ICHECK(w_->devices.size() != 0) << "No SYCL device";
    // get kernel
    void (*kernel_func)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) = (void (*)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args))dlsym(so_handler_, func_name_.c_str());
    ICHECK(kernel_func != NULL) << "ERROR:"<<dlerror()<<":dlsym\n";
    // get thread dimension
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    sycl::range<3> k0_dimGrid(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    sycl::range<3> k0_dimBlock(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
    //printf("%zu\t%zu\t%zu\n", k0_dimGrid.get(0), k0_dimGrid.get(1), k0_dimGrid.get(2));
    //printf("%zu\t%zu\t%zu\n", k0_dimBlock.get(0), k0_dimBlock.get(1), k0_dimBlock.get(2));
    //std::cout << func_name_<< std::endl;
    syclT::SYCLThreadEntry* t = w_->GetThreadEntry();
    sycl::queue Queue = w_->GetQueue(t->device);
    if (w_->IsProfiling(t->device)){
      w_->GetEventQueue(t->device).resize(w_->GetEventQueue(t->device).size() + 1);
      LOG(WARNING) << "todo, not support now";
    }else{
      SYCL_CALL(kernel_func(Queue, k0_dimGrid, k0_dimBlock, void_args));
    }
  }

 private:
  // global workspace.
  syclT::SYCLWorkspace* w_;
  // The module
  SYCLModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // launch parameters config
  LaunchParamConfig launch_param_config_;
  // share library handler
  void *so_handler_;
};

SYCLModuleNode::~SYCLModuleNode() {
  {
    // free the kernel ids in global table.
    std::lock_guard<std::mutex> lock(workspace_->mu);
    for (auto& kv : kid_map_) {
      workspace_->free_kernel_ids.push_back(kv.second.kernel_id);
    }
    if(so_handler_ != nullptr){
      dlclose(so_handler_);
    }
  }
  // free the kernels
}

syclT::SYCLWorkspace* SYCLModuleNode::GetGlobalWorkspace() {
  return syclT::SYCLWorkspace::Global();
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
  if(so_handler_ == nullptr){
    std::string sharedlibpath = "/tmp/tvm_sycl/kernels.so";
    so_handler_ = dlopen(sharedlibpath.c_str(), RTLD_LAZY);
    ICHECK(so_handler_ != NULL) << "ERROR:"<<dlerror()<<":dlopen\n";
  }
  VLOG(0) << "SYCLModuleNode::begin initialize the wrapped func:";
  // initialize the wrapped func.
  f.Init(this, name, info.arg_types.size(), info.launch_param_tags, so_handler_);
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
  if (fmt_ == "sycl") {
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
  parsed_kernels_ = SplitKernels(GetSource("sycl"));
  ICHECK(!parsed_kernels_.empty()) << "The SYCL module expects a kernel delimited "
                                   << "source from code generation, but no kernel "
                                   << "delimiter was found.";
  ICHECK(fmap_.size() == parsed_kernels_.size())
      << "The number of parsed kernel sources does not match the number of kernel functions";
  // compile kernels .so
  if(access("/tmp/tvm_sycl", F_OK) == -1){
    system("mkdir /tmp/tvm_sycl");
  }
  std::string filepath = "/tmp/tvm_sycl/kernels.cpp";
  std::string sharedlibpath = "/tmp/tvm_sycl/kernels.so";
  std::ofstream kernels_file;
  kernels_file.open(filepath);
  kernels_file << GetSource("sycl");
  kernels_file.close();
  std::string cmd = "$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fPIC -shared "+filepath+" -o "+sharedlibpath;
  system(cmd.c_str());
  VLOG(0) << cmd;
  /*
  for (auto& kv : parsed_kernels_) {
    std::string kernel_name = kv.first;
    std::string kernel_code = kv.second;
    std::string filepath = "/tmp/tvm_sycl/"+kernel_name+".cpp";
    std::string sharedlibpath = "/tmp/tvm_sycl/"+kernel_name+".so";
    std::ofstream myfile;
    myfile.open(filepath);
    myfile << kernel_code;
    myfile.close();
    std::string cmd = "$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fPIC -shared " + filepath + " -o " + sharedlibpath;
    system(cmd.c_str());
    VLOG(0) << cmd;
  }*/
}

Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source);
  n->Init();
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
