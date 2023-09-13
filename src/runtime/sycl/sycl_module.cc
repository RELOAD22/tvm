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
#include "../file_utils.h"
#include "sycl_common.h"

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
    // swap the first(0) and third(2) dimension size in a SYCL workGroup. Because the max size of first dimension size in SYCL is small
    // (for example, 64 in A100), the schedule generated by tvm often bigger than it.
    sycl::range<3> k0_dimBlock(wl.block_dim(2), wl.block_dim(1), wl.block_dim(0));
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
    Queue.wait_and_throw();
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
    // close share library handler
    if(so_handler_ != nullptr){
      dlclose(so_handler_);
    }
    // delete source code and share library
    std::remove(this->lib_compiler.source_file_path.c_str());
    std::remove(this->lib_compiler.shared_lib_path.c_str());
  }
  // free the kernels
}

syclT::SYCLWorkspace* SYCLModuleNode::GetGlobalWorkspace() {
  return syclT::SYCLWorkspace::Global();
}

std::pair<int, std::string> shell_exec(std::string command, bool with_stderr = true) {
  char buffer[1024];
  std::string result = "";
  if (with_stderr)
    command += " 2>&1";
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) 
    return {errno, "popen failed."};
  while (!feof(pipe)) {
    // use buffer to read and add to result
    if (fgets(buffer, sizeof(buffer), pipe) != NULL)
      result += buffer;
  }
  pclose(pipe);
  return {0, result};
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
  f.Init(this, name, info.arg_types.size(), info.launch_param_tags, so_handler_);
  VLOG(0) << "SYCLModuleNode::finish initialize the wrapped func:";
  return PackFuncVoidAddr(f, info.arg_types);
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

  // split into source artifacts for each kernel
  parsed_kernels_ = SplitKernels(GetSource("sycl"));
  ICHECK(!parsed_kernels_.empty()) << "The SYCL module expects a kernel delimited "
                                   << "source from code generation, but no kernel "
                                   << "delimiter was found.";
  ICHECK(fmap_.size() == parsed_kernels_.size())
      << "The number of parsed kernel sources does not match the number of kernel functions";
  if(this->lib_compiler.load_from_file){
    so_handler_ = dlopen(this->lib_compiler.shared_lib_path.c_str(), RTLD_LAZY);
  }else{
    // create the folder to store sycl temporary files
    if(access(this->lib_compiler.prefix.c_str(), F_OK) == -1){
      std::string cmd = "mkdir "+this->lib_compiler.prefix;
      system(cmd.c_str());
    }
    // sycl kernel source code
    std::ofstream kernels_file;
    kernels_file.open(this->lib_compiler.source_file_path);
    kernels_file << GetSource("sycl");
    kernels_file.close();
    // compile kernel source code to share libary
    std::cout<<"[SYCL] Compile kernels source code(" + this->lib_compiler.source_file_path + ") to share library."<<std::endl;
    VLOG(0) << this->lib_compiler.command;
    system(this->lib_compiler.command.c_str());
    /*std::string exec_result = shell_exec(this->lib_compiler.command).second;
    std::cout<< exec_result;*/
    // dlopen sycl share libary
    so_handler_ = dlopen(this->lib_compiler.shared_lib_path.c_str(), RTLD_LAZY);
  }
  ICHECK(so_handler_ != NULL) << "ERROR:"<<dlerror()<<":dlopen\n";
}

Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}
Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source, std::string path_key) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source, path_key);
  n->Init();
  return Module(n);
}

void SYCLModuleNode::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, data_);
  SaveBinaryToFile(file_name+"_path_key", this->lib_compiler.file_path_key);
}

void SYCLModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
  stream->Write(this->lib_compiler.file_path_key);
}

// Load module from module.
Module SYCLModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  std::string path_key;
  LoadBinaryFromFile(file_name+"_path_key", &path_key);
  return SYCLModuleCreate(data, fmt, fmap, std::string(), path_key);
}

Module SYCLModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  std::string path_key;
  stream->Read(&path_key);
  return SYCLModuleCreate(data, fmt, fmap, std::string(), path_key);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_sycl").set_body_typed(SYCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_syclbin").set_body_typed(SYCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_sycl").set_body_typed(SYCLModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
