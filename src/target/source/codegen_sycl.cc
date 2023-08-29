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

#include "../../runtime/sycl/sycl_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

CodeGenSYCL::CodeGenSYCL() {
  // Set SYCL specific restrict keyword
  restrict_keyword_ = "";
}


void CodeGenSYCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

void CodeGenSYCL::AddFunction(const PrimFunc& f) {
  // from CodeGenC -------------------
  // clear previous generated state.
  this->InitFuncState(f);
  // clear pre mem_allocate_code
  this->mem_allocate_code = std::ostringstream("");
  // reserve keywords
  ReserveKeywordsAsUnique();
  this->PrintExtraAttrs(f);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenSYCL: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  // Print the packed function
  stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  stream << "// Function: " << static_cast<std::string>(global_symbol.value()) << "\n";
  PrintFuncPrefix();
  stream << " " << static_cast<std::string>(global_symbol.value());
  stream << "(queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {\n";
  int func_scope = this->BeginScope();
  // Print the packed function's void_args
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
    this->PrintIndent();
    PrintType(GetType(v), stream);
    if (no_alias) {
      PrintRestrict(v, stream);
    }
    stream << ' ' << vid << " = (";
    PrintType(GetType(v), stream);
    stream << ")(*(int64_t *)(void_args[" << i << "]));\n";
  }

  // Print submit  
  this->PrintIndent();
  stream << "Q.submit([&](handler &h) {\n";
  int func_scope2 = this->BeginScope();
  
  this->PrintIndent();
  stream << "h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {\n";
  int func_scope3 = this->BeginScope();
  // Function code
  this->PrintStmt(f->body);
  this->EndScope(func_scope3);
  this->PrintIndent();
  stream << "});\n";
  this->EndScope(func_scope2);
  this->PrintIndent();
  stream << "});\n";
  this->PrintIndent();
  stream << "Q.wait();\n";
  this->EndScope(func_scope);
  stream << "}\n";
  /*
  // insert mem_allocate_code
  std::string preCode = stream.str();
  size_t start = preCode.find("Q.submit([&](handler &h) {\n");
  start += strlen("Q.submit([&](handler &h) {\n");
  preCode.insert(start, mem_allocate_code.str());
  stream = std::ostringstream(preCode);
  */
}

void CodeGenSYCL::PrintFuncPrefix() {
  stream << "#ifdef __cplusplus\n"
         << "extern \"C\"\n"
         << "#endif\n"
         << "void";
}

std::string CodeGenSYCL::Finish() {
  // fp16 and fp64
  if (enable_fp16_) {
  }

  if (enable_fp64_) {
  }

  // atomic_add
  if (enable_atomics_) {

  }
  return CodeGenC::Finish();
}

void CodeGenSYCL::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    // swap the first(0) and third(2) dimension size in a SYCL workGroup.
    os << "item.get_local_id(" << 2-ts.dim_index << ")";
  } else {
    os << "item.get_group(" << ts.dim_index << ")";
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
  LOG(FATAL) << "Cannot convert type " << t << " to SYCL type";
}

void CodeGenSYCL::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
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
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer_var) << " + ";
  PrintExpr(base, os);
}

void CodeGenSYCL::PrintStorageSpace(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  os << "sycl::access::address_space::generic_space";
  /*
  if (scope == "global") {
    os << "sycl::access::address_space::global_space";
  } else if (scope == "shared") {
    os << "sycl::access::address_space::local_space";
  } else {
    os << "sycl::access::address_space::private_space";
  }
  */
}

std::string CodeGenSYCL::GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) {
  std::ostringstream os;
  os << "({";
  os << "vec<";
  PrintType(t.element_of(), os);
  os << ", " << t.lanes() <<"> x; ";
  os << "x.load(0, multi_ptr<";
  PrintType(t.element_of(), os);
  os << ", ";
  std::string scope = "global";
  auto it = alloc_storage_scope_.find(buffer->data.get());
  if(it != alloc_storage_scope_.end()){
    scope = it->second;
  }
  PrintStorageSpace(scope, os);
  os << ", sycl::access::decorated::no";
  os << ">(";
  PrintVecAddr(buffer, t, base, os);
  os << "));";
  os << "x;";
  os << "})";
  return os.str();
}

void CodeGenSYCL::PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << value << ".store(0, multi_ptr<";
  PrintType(t.element_of(), stream);
  stream << ", ";
  std::string scope = "global";
  auto it = alloc_storage_scope_.find(buffer->data.get());
  if(it != alloc_storage_scope_.end()){
    scope = it->second;
  }
  PrintStorageSpace(scope, stream);
  stream  << ", sycl::access::decorated::no";
  stream  << ">(";
  PrintVecAddr(buffer, t, base, stream);
  stream << "));\n";
}

void CodeGenSYCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "group_barrier(item.get_sub_group());\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "group_barrier(item.get_group());\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier is not supported";
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
    os << value << ".convert<";
    PrintType(target.element_of(), os);
    os << ">()";
    os << ")";
  }
  return os.str();
}

void CodeGenSYCL::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->ConstantAllocationSize() * op->dtype.lanes()});
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting SYCL. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    this->PrintIndent();
    stream<< "auto " << vid << " = *sycl::ext::oneapi::group_local_memory<";
    PrintType(op->dtype, stream);
    stream << '[' << constant_size << "]>(item.get_group());\n";
    /*
    for (int i = 0; i < 4; ++i) {
      mem_allocate_code << ' ';
    }
    mem_allocate_code << "sycl::local_accessor<";
    PrintType(op->dtype.element_of(), mem_allocate_code);
    mem_allocate_code << ", 1> "<< vid << "(sycl::range(" << constant_size<< "),h);\n";
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
    ICHECK_EQ(load->indices.size(), 1) << "CodeGenSYCL only supports flat memory allocations.";
    os << "((";
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer->data.get()) << " + ";
    this->PrintExpr(load->indices[0], os);
    os << ')';
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
    // default generated math function name is non-exist in SYCL. Not know where the function names are generated ?
    std::unordered_map<std::string, std::string> func_names = {{"expf", "exp"}, {"powf", "pow"}, {"tanhf", "tanh"}};
    if(func_names.find(func->value) != func_names.end()){
      func = StringImm(func_names[func->value]);
    }
    this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenSYCL::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "(";
  os << "vec<";
  PrintType(op->dtype.element_of(), os);
  os << ", " << op->lanes << ">";
  os << "{";
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

void CodeGenSYCL::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << "[" << std::hex << i << "]" << std::dec;
}

void CodeGenSYCL::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << "[" << std::hex << i << "]" << " = " << value << ";\n" << std::dec;
}

void CodeGenSYCL::PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  /*
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }
  */

  if (i == 0) {
    os << "((";
    PrintType(t, os);
    os << "){";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "})";
  }
  return;
}

void CodeGenSYCL::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "((int" << op->lanes << "){";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "})";
}

runtime::Module BuildSYCL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  std::stringstream code;
  code << "// tvm target: " << target->str() << "\n";
  code << "#include <CL/sycl.hpp>\n";
  code << "using namespace sycl;\n";

  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    CodeGenSYCL cg;
    cg.Init(output_ssa);
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    // Debug for SYCL
    VLOG(0) << "BuildSYCL: code:\n" << fsource;
    if (fpostproc) {
      fsource = (*fpostproc)(fsource).operator std::string();
    }
    code << fsource;
  }

  return SYCLModuleCreate(code.str(), "sycl", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.sycl").set_body_typed(BuildSYCL);
}  // namespace codegen
}  // namespace tvm
