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
 * \file codegen_sycl.h
 * \brief Generate SYCL device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_SYCL_H_
#define TVM_TARGET_SOURCE_CODEGEN_SYCL_H_

#include <tvm/target/codegen.h>

#include <string>
#include <unordered_map>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenSYCL final : public CodeGenC {
 public:
  CodeGenSYCL();
  std::string Finish();

  void AddFunction(const PrimFunc& f);
  // override print thread tag.
  void InitFuncState(const PrimFunc& f) final;
  void PrintFuncPrefix() final;                                              // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;                             // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintStorageSync(const CallNode* op) final;                           // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;                        // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os) final;                  // NOLINT(*)
  void PrintStorageSpace(const std::string& scope, std::ostream& os);
  std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) final;
  void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                     const std::string& value) final;  // NOLINT(*)
  // the address of load/store
  void PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                    std::ostream& os);                                           // NOLINT(*)
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DataType t, int i, const std::string& value) final;
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) final;
  std::string CastFromTo(std::string value, DataType from, DataType target);     // NOLINT(*)
  std::string CastTo(std::string value, DataType target);                        // NOLINT(*)

  // overload visitor
  void VisitStmt_(const AllocateNode* op) final;                     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;   // NOLINT(*)

  // overload min and max to avoid ambiguous call errors
  void VisitExpr_(const MinNode* op, std::ostream& os) final;
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;
  void VisitExpr_(const AndNode* op, std::ostream& os) final;
  void VisitExpr_(const OrNode* op, std::ostream& os) final;
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;
  void VisitExpr_(const RampNode* op, std::ostream& os) final;

 private:
  // whether enable fp16 and fp64 extension
  bool enable_fp16_{false};
  bool enable_fp64_{false};
  // Whether to enable atomics extension.
  bool enable_atomics_{false};
  // Mapping from buffer to allocation size.
  // Useful to track when a scalar store of a vectorized texture load is required.
  std::unordered_map<const Object*, size_t> allocation_size_;
  //local memory allocate code
  std::ostringstream mem_allocate_code;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_SYCL_H_
