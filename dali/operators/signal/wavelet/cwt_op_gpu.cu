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

#include <memory>
#include <utility>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/signal/wavelet/cwt_args.h"
#include "dali/kernels/signal/wavelet/cwt_gpu.h"
#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"
#include "dali/operators/signal/wavelet/cwt_op.h"
#include "dali/operators/signal/wavelet/wavelet_name.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Cwt)
    .DocStr(R"code(Calculates continuous wavelet transform (CWT).

Result values of transform are computed for all specified scales. 
)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumInputs)
    .AddArg("a", R"code(List of scale coefficients)code", DALIDataType::DALI_FLOAT_VEC)
    .AddArg("wavelet", R"code(Type of mother wavelet)code", DALIDataType::DALI_WAVELET_NAME)
    .AddOptionalArg(
        "args", R"code(Additional arguments to mother wavelet. They may be declared for wavelets:
- w1
- w2
- w3
)code",
        std::vector<float>());

template <typename T>
struct CwtImplGPU : public OpImplBase<GPUBackend> {
 public:
  using CwtArgs = kernels::signal::CwtArgs<T>;
  using CwtKernel = kernels::signal::CwtGpu<T>;

  explicit CwtImplGPU(CwtArgs args) : args_(std::move(args)) {
    kmgr_cwt_.Resize<CwtKernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  CwtOpArgs<float> args_;

  kernels::KernelManager kmgr_cwt_;
  kernels::KernelManager kmgr_wvlt_;

  std::vector<OutputDesc> cwt_out_desc_;
  TensorList<GPUBackend> cwt_out_;
};

template <typename T>
bool CwtImplGPU<T>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_view = view<const T>(input);

  auto type = type2id<T>::value;

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  // here setup wavelet kernel

  auto &req = kmgr_cwt_.Setup<CwtKernel>(0, ctx, in_view);
  output_desc.resize(1);
  output_desc[0].type = type;
  output_desc[0].shape = req.output_shapes[0];  // TODO: list of a x nsamples tensors

  // here setup sum (reduce) kernel

  return true;
}

template <typename T>
void CwtImplGPU<T>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  auto in_view = view<const T>(input);
  auto out_view = view<T>(output);

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  // here run wavelet kernel

  kmgr_cwt_.Run<CwtKernel>(0, ctx, out_view, in_view, args_);

  // here sum (reduce)
}

template <>
bool Cwt<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  const auto &input = ws.Input<GPUBackend>(0);
  auto type = input.type();

  TYPE_SWITCH(type, type2id, T, (float), (
      using Impl = CwtImplGPU<T>;
      if (!impl_ || type != type_) {
        impl_ = std::make_unique<Impl>(args_);
        type_ = type;
      }
  ), DALI_FAIL(make_string("Unsupported data type: ", type)));  // NOLINT

  impl_->SetupImpl(output_desc, ws);
  return true;
}

template <>
void Cwt<GPUBackend>::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Cwt, Cwt<GPUBackend>, GPU);

}  // namespace dali
