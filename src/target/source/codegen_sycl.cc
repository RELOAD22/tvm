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
 * \file codegen_sycl.cc
 */
#include "codegen_sycl.h"

#include <cmath>
#include <string>
#include <vector>
#include <regex>

#include "../../runtime/sycl/sycl_module.h"
#include "../../runtime/opencl/opencl_module.h"
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

class InferTextureAccess : public StmtExprVisitor {
 public:
  static constexpr const uint8_t kReadAccess = 1;
  static constexpr const uint8_t kWriteAccess = 2;

  InferTextureAccess() {}
  std::unordered_map<const VarNode*, std::string> Infer(const Stmt& n) {
    StmtExprVisitor::VisitStmt(n);
    std::unordered_map<const VarNode*, std::string> storage_scope_qualifiers;
    for (auto& texture : var_access_map_) {
      if (texture.second == kReadAccess) {
        storage_scope_qualifiers.insert({texture.first, "texture_read"});
      } else if (texture.second == kWriteAccess) {
        storage_scope_qualifiers.insert({texture.first, "texture_write"});
      } else if (texture.second == (kReadAccess | kWriteAccess)) {
        storage_scope_qualifiers.insert({texture.first, ""});
      }
    }
    return storage_scope_qualifiers;
  }
  void VisitExpr_(const CallNode* op) {
    if (op->op.same_as(builtin::texture2d_load())) {
      var_access_map_[op->args[0].as<VarNode>()] |= kReadAccess;
    } else if (op->op.same_as(builtin::texture2d_store())) {
      var_access_map_[op->args[0].as<VarNode>()] |= kWriteAccess;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

 private:
  std::unordered_map<const VarNode*, uint8_t> var_access_map_;
};

CodeGenSYCL::CodeGenSYCL() {
  // Set OpenCL specific restrict keyword
  restrict_keyword_ = "restrict";
}

void CodeGenSYCL::Init(bool output_ssa, bool emit_asserts, std::string target_str,
                        const std::unordered_set<std::string>& devices) {
  //emit_asserts_ = emit_asserts;
  //declared_globals_.clear();
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#define TVM_EXPORTS\n";

  decl_stream << "#include <CL/sycl.hpp>\n";
  decl_stream << "using namespace sycl;\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenSYCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  this->SetTextureScope(InferTextureAccess().Infer(f->body));
  for (Var arg : f->params) {
    auto ptr_type = arg->type_annotation.as<PointerTypeNode>();
    if (ptr_type && runtime::IsTextureStorage(std::string(ptr_type->storage_scope))) {
      // Storage scope qualifiers for textures are inferred
      // and set prior to function codegen.
      continue;
    } else if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}



void CodeGenSYCL::AddFunction(const PrimFunc& f) {
  //function_names_.push_back(global_symbol.value());

  // from CodeGenC -------------------
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenSYCL: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->stream << "#define";
  this->PrintExtraAttrs(f);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) 
              << "_dev() ";
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (v.dtype().is_handle()) {
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    }
  }            
  stream << " {\\\n";
  //
  std::ostringstream preStream;
  preStream << this->stream.str();
  this->stream.str("");
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();

  //替换#define区域内\n为\\n, 因为#define需要在同一行
  std::string defineStr = std::regex_replace(this->stream.str(), std::regex("\n"), "\\\n");
  this->stream.str("");
  this->stream<< preStream.str() << defineStr;
  this->stream << "}\n\n";
  // --------------------


  // Print the packed function
  stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  PrintFuncPrefix();
  stream << " " << static_cast<std::string>(global_symbol.value());
  stream << "(queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) {\n";

  // Print the packed function's void_args
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = GetVarID(v.get());
    stream << "  ";
    PrintType(GetType(v), stream);
    stream << ' ' << vid << " = (";
    PrintType(GetType(v), stream);
    stream << ")(*(int64_t *)(void_args[" << i << "]));\n";
  }

  // Print submit  
  stream << "  Q.submit([&](handler &h) {\n";
  //stream <<"    "<< this->mem_allocate_code.str();
  stream << "    h.parallel_for(sycl::nd_range<3>(k0_dimGrid * k0_dimBlock, k0_dimBlock), [=](sycl::nd_item<3> item_ct1) {\n";
  stream << "      " << global_symbol.value() << "_dev"
         << "();\n";
  stream << "    }); });\n";   


  stream << "  Q.wait();\n";
  stream << "}\n";

}

void CodeGenSYCL::PrintFuncPrefix() {
  stream << "#ifdef __cplusplus\n"
         << "extern \"C\"\n"
         << "#endif\n"
         << "void";
}

std::string CodeGenSYCL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream << "#ifdef cl_khr_fp16\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                   "#elif defined(cl_amd_fp16)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
                   "#else\n"
                   "#error \"Half precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream << "#ifdef cl_khr_fp64\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                   "#elif defined(cl_amd_fp64)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
                   "#else\n"
                   "#error \"Double precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  // Enable atomic_add used by get_valid_counts. Only needed for OpenCL < 1.1.
  if (enable_atomics_) {
    decl_stream << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                   "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n\n";
  }

  // Enable OpenCL 1.2 sampler-less texture reads, but utilize
  // provided sampler in OpenCL 2.0.
  if (enable_compliant_texture_reads_) {
    // TODO(csullivan, lunderberg): Extend device attribute querying to support remote devices
    // generically through the device API such that a target can be created from a specific device's
    // attributes and utilized during codegen. Potential generlization of #8127 (c02cafb) for remote
    // devices.
    //
    // E.g. Only provide an image sampler when the local or remote device supports OpenCL 2.0,
    //      see below for context.
    //
    // For backwards compatibility with OpenCL 1.2, sampler-less read_image calls are used.
    // By default in sampler-less read_image calls OpenCL defaults to
    // sampler_ = "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST";
    // See section 6.12.14.3 Built-in Image Sampler-less Read Functions in the OpenCL 1.2
    // specification. For OpenCL 2.0 it can be preferable to use,
    // sampler_ = "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST";
    // For now we rely on OpenCL preprocessor directives to utilize the correct behavior
    // depending on the OpenCL version detected at OpenCL compile time.
    decl_stream << "#ifdef __OPENCL_VERSION__\n"
                << "#if __OPENCL_VERSION__ == CL_VERSION_2_0\n"
                << "#define READ_IMAGEH(image, sampler, coord) "
                << "read_imageh(image, sampler, coord)\n"
                << "#define READ_IMAGEF(image, sampler, coord) "
                << "read_imagef(image, sampler, coord)\n"
                << "#else\n"
                << "#define READ_IMAGEH(image, sampler, coord) "
                << "read_imageh(image, coord)\n"
                << "#define READ_IMAGEF(image, sampler, coord) "
                << "read_imagef(image, coord)\n"
                << "#endif\n"
                << "#endif\n\n";
  }
  return CodeGenC::Finish();
}

void CodeGenSYCL::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "item_ct1.get_local_id(" << ts.dim_index << ")";
  } else {
    os << "item_ct1.get_group(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::UInt(64), iv->var.dtype());
}

void CodeGenSYCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        enable_fp16_ = true;
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        enable_fp64_ = true;
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 1:
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to OpenCL type";
}

void CodeGenSYCL::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    if (runtime::IsTextureStorage(std::string(ptr->storage_scope))) {
      os << "image2d_t";
    } else {
      PrintType(ptr->element_type, os);
      os << '*';
    }
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

void CodeGenSYCL::PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                                 std::ostream& os) {  // NOLINT(*)
  const VarNode* buffer_var = buffer->data.get();
  if (!HandleTypeMatch(buffer_var, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer_var);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer_var) << " + ";
  PrintExpr(base, os);
}
std::string CodeGenSYCL::GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) {
  std::ostringstream os;

  std::ostringstream x;
  PrintType(t, x);
  std::string type = x.str();
  std::string basic_type = type;
  if(t.lanes() >= 2){
    basic_type = std::regex_replace(type, std::regex("\\d+"), "");
  }
  //for example, type=float4, basic_type=float

  os << "({";
  os << "vec<"<<basic_type<<", " << t.lanes() <<"> x; ";
  os << "x.load(0, multi_ptr<"<<basic_type<<", sycl::access::address_space::global_space>(";
  PrintVecAddr(buffer, t, base, os);
  os << ")); ";
  os << "x;";
  os << "})";
  return os.str();
}

void CodeGenSYCL::PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << value << ".store(0, multi_ptr<float, sycl::access::address_space::global_space>(";
  PrintVecAddr(buffer, t, base, stream);
  stream << "));\n";
}

void CodeGenSYCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "group_barrier(item_ct1.get_group());\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "group_barrier(item_ct1.get_group());\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenSYCL::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if(scope =="shared"){
    std::string code = static_cast<std::ostringstream&>(os).str();
    std::string str = code.substr(code.length()-10, std::string::npos);
    if(str!="(volatile "){
      std::cout<<str<<std::endl;
    }
  }
  //ICHECK_NE(scope, "shared") << "no storage scope keyword in SYCL!";
/*
  if (scope == "global") {
    os << "__global ";
  } else if (scope == "shared") {
    os << "__local ";
  } else if (scope == "texture_read") {
    os << "__read_only ";
  } else if (scope == "texture_write") {
    os << "__write_only ";
  }*/
}

void CodeGenSYCL::PrintRestrict(const Var& v, std::ostream& os) {
  /*
  // Apply restrict qualifer for non-texture types only
  if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
    if (!runtime::IsTextureStorage(std::string(ptr->storage_scope))) {
      os << ' ' << restrict_keyword_;
    }
  }*/
}

std::string CodeGenSYCL::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  return CastTo(value, target);
}

std::string CodeGenSYCL::CastTo(std::string value, DataType target) {
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
  }
  return os.str();
}

void CodeGenSYCL::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
}

void CodeGenSYCL::VisitStmt_(const BufferStoreNode* op) {
  if (auto call = op->value.as<CallNode>()) {
    if (call->op.same_as(builtin::texture2d_load())) {
      need_texture_ssa_ = false;
      // If storing a texture load into a buffer, don't use an
      // intermediate local unless the buffer allocation is a
      // single element selected from the texture read.
      auto it = allocation_size_.find(op->buffer->data.get());
      if (it != allocation_size_.end() && it->second == 1) {
        need_texture_ssa_ = true;
      }
    }
  }
  CodeGenC::VisitStmt_(op);
  need_texture_ssa_ = true;
}

void CodeGenSYCL::VisitExpr_(const CastNode* op, std::ostream& os) {
  if (auto call = op->value.as<CallNode>()) {
    if (call->op.same_as(builtin::texture2d_load())) {
      need_texture_ssa_ = false;
    }
  }
  CodeGenC::VisitExpr_(op, os);
  need_texture_ssa_ = true;
}

void CodeGenSYCL::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->ConstantAllocationSize() * op->dtype.lanes()});
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  ICHECK_NE(scope, "global") << "Cannot allocate global memory in kernel when targeting SYCL. ";
  if (scope == "shared") {
    this->PrintIndent();
    stream<< "auto " << vid << " = *sycl::ext::oneapi::group_local_memory<";
    PrintType(op->dtype, stream);
    stream << '[' << constant_size << "]>(item_ct1.get_group());\n";
    /*
    this->mem_allocate_code << "sycl::local_accessor<";
    PrintType(op->dtype, this->mem_allocate_code);
    this->mem_allocate_code << ", 1> "<< vid << "(sycl::range(" << constant_size<< "),h);\n";
    */
  }else{
    this->PrintIndent();
    PrintType(op->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenSYCL::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1) << "CodeGenOpenCL only supports flat memory allocations.";
    os << "((";
    auto it = alloc_storage_scope_.find(load->buffer->data.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer->data.get()) << " + ";
    this->PrintExpr(load->indices[0], os);
    os << ')';
  } else if (op->op.same_as(builtin::texture2d_store())) {
    auto* ptr_type = op->args[0].as<VarNode>()->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type != nullptr) << "Texture Var's must be of PointerType";
    ICHECK(runtime::IsTextureStorage(std::string(ptr_type->storage_scope)))
        << "builtin::texture2d_store() only supports storing to texture buffers";
    DataType buffer_type = ptr_type->element_type.as<PrimTypeNode>()->dtype;
    if (buffer_type.is_float16()) {
      os << "write_imageh(";
    } else if (buffer_type.is_float()) {
      os << "write_imagef(";
    } else {
      LOG(FATAL) << "Unsupported type: " << buffer_type
                 << ", currently only float and half are supported for image2d OpenCL codegen.";
    }
    this->PrintExpr(op->args[0], os);
    os << ", ";
    os << "(int2)(";
    this->PrintExpr(op->args[1], os);
    os << ", ";
    this->PrintExpr(op->args[2], os);
    os << "), ";
    this->PrintExpr(op->args[3], os);
    os << ")";
  } else if (op->op.same_as(builtin::texture2d_load())) {
    enable_compliant_texture_reads_ = true;
    std::stringstream ss;
    if (op->dtype.is_float16()) {
      ss << "READ_IMAGEH(";
    } else if (op->dtype.is_float()) {
      ss << "READ_IMAGEF(";
    } else {
      LOG(FATAL) << "Unsupported type: " << op->dtype
                 << ", currently only float and half are supported for image2d OpenCL codegen.";
    }
    this->PrintExpr(op->args[0], ss);
    ss << ", ";
    ss << "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, ";
    ss << "((int2)(";
    this->PrintExpr(op->args[1], ss);
    ss << ", ";
    this->PrintExpr(op->args[2], ss);
    ss << ")))";

    // Only use local SSA if texture is not already being stored
    if (need_texture_ssa_) {
      std::string rhs = SSAGetID(ss.str(), op->dtype.with_lanes(4));
      if (op->args.back().as<RampNode>()) {
        os << rhs;
      } else {
        os << "((";
        this->PrintType(op->dtype.with_lanes(1), os);
        os << "*)&" << rhs << ")[";
        this->PrintExpr(op->args.back(), os);
        os << "]";
      }
    } else {
      os << ss.str();
    }
  } else if (op->op.same_as(builtin_call_extern_)) {
    auto func = Downcast<StringImm>(op->args[0]);
    // Enable atomics extension if used.
    if (func->value == "atomic_add") {
      enable_atomics_ = true;
    }
    CodeGenC::VisitExpr_(op, os);
  } else if (op->op.same_as(builtin_call_pure_extern_)) {
      ICHECK_GE(op->args.size(), 1U);
      auto func = Downcast<StringImm>(op->args[0]);
      if(func->value == "expf"){
        func = StringImm("exp");
      }else if(func->value == "powf"){
        func = StringImm("pow");
      }
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenSYCL::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->dtype, os);
  os << "){";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "})";
}

void CodeGenSYCL::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      os << "-";
    }
    os << "INFINITY";
  } else if (std::isnan(op->value)) {
    os << "NAN";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr, std::ostream& os, CodeGenSYCL* p) {
  if (op->dtype.lanes() == 1) {
    os << opstr << "((";
    p->PrintType(op->a->dtype, os);
    os << ")";
    p->PrintExpr(op->a, os);
    os << ", (";
    p->PrintType(op->b->dtype, os);
    os << ")";
    p->PrintExpr(op->b, os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenSYCL::VisitExpr_(const MinNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "min", os, this);
}

void CodeGenSYCL::VisitExpr_(const MaxNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "max", os, this);
}

void CodeGenSYCL::VisitExpr_(const AndNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " && ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenSYCL::VisitExpr_(const OrNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " || ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenSYCL::VisitExpr_(const SelectNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "select(";
  PrintExpr(op->false_value, oss);
  os << CastFromTo(oss.str(), op->false_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->true_value, oss);
  os << CastFromTo(oss.str(), op->true_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->condition, oss);
  if (op->dtype.is_float()) {
    if (op->condition.dtype().is_uint() || op->condition.dtype().is_int()) {
      os << oss.str();
    } else {
      os << CastTo(oss.str(), DataType::Int(op->dtype.bits(), op->dtype.lanes()));
    }
  } else {
    os << CastFromTo(oss.str(), op->condition.dtype(), op->dtype);
  }
  os << ")";
}

void CodeGenSYCL::SetTextureScope(
    const std::unordered_map<const VarNode*, std::string>& scope) {  // NOLINT(*)
  for (auto& texture : scope) {
    alloc_storage_scope_.insert(texture);
  }
}

runtime::Module BuildSYCL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenSYCL cg;
    bool emit_asserts = false;

    std::unordered_set<std::string> devices;
    if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts") != nullptr) {
      Map<GlobalVar, String> device_contexts =
          mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
      for (auto const& context : device_contexts) {
        devices.insert(context.second.data());
      }
    }
    cg.Init(output_ssa, emit_asserts, target->str(), devices);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    // Debug for SYCL
    VLOG(0) << "BuildSYCL: code:\n" << fsource;
    if (fpostproc) {
      fsource = (*fpostproc)(fsource).operator std::string();
    }
    code << fsource;
  }

  return SYCLModuleCreate(code.str(), "cl", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.sycl").set_body_typed(BuildSYCL);
}  // namespace codegen
}  // namespace tvm
