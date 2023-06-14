// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_SIGNAL_WAVELET_CWT_OP_H_
#define DALI_OPERATORS_SIGNAL_WAVELET_CWT_OP_H_

#include <memory>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/signal/wavelet/cwt_args.h"
#include "dali/operators/signal/wavelet/cwt_op_args.h"
#include "dali/operators/signal/wavelet/wavelet_name.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

template <typename Backend>
class Cwt : public Operator<Backend> {
 public:
  explicit Cwt(const OpSpec &spec) : Operator<Backend>(spec) {
    if (!spec.HasArgument("a")) {
      DALI_ENFORCE("`a` argument must be provided.");
    }
    args_.a = spec.GetArgument<std::vector<float>>("a");

    if (!spec.HasArgument("wavelet")) {
      DALI_ENFORCE("`wavelet` argument must be provided.");
    }
    args_.wavelet = spec.GetArgument<DALIWaveletName>("wavelet");
  }

 protected:
  bool CanInferOutputs() const override {
    return true;
  }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  kernels::KernelManager kmgr_;
  CwtOpArgs<float> args_;

  std::unique_ptr<OpImplBase<Backend>> impl_;
  DALIDataType type_ = DALI_NO_TYPE;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SIGNAL_WAVELET_CWT_OP_H_
