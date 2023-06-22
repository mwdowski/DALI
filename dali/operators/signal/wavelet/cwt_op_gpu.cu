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
#include "dali/core/tensor_shape.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/signal/wavelet/cwt_args.h"
#include "dali/kernels/signal/wavelet/cwt_gpu.h"
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"
#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"
#include "dali/operators/signal/wavelet/cwt_op.h"
#include "dali/operators/signal/wavelet/cwt_op_args.h"
#include "dali/operators/signal/wavelet/wavelet_name.h"
#include "dali/operators/signal/wavelet/wavelet_run.h"
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
    .AddArg("wavelet_args",
            R"code(Additional arguments to mother wavelet. They may be declared for wavelets:
- w1 TODO: fill in for each wavelet that needs arguments
- w2
- w3
)code",
            DALIDataType::DALI_FLOAT_VEC);

template <typename T>
struct CwtImplGPU : public OpImplBase<GPUBackend> {
 public:
  using CwtKernel = kernels::signal::CwtGpu<T>;
  using WvltKernel =
      kernels::signal::WaveletGpu<T, kernels::signal::MexicanHatWavelet>;  // TODO:
                                                                           // change it
  using CwtArgs = kernels::signal::CwtArgs<T>;
  using WvltSpan = kernels::signal::WaveletSpan<T>;

  explicit CwtImplGPU(CwtOpArgs<float> args) : args_(std::move(args)) {
    kmgr_cwt_.Resize<CwtKernel>(1);
    kmgr_wvlt_.Resize<WvltKernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  TensorListShape<> GetWaveletOutputShape(const size_t &a_size, const WvltSpan &span);
  CwtOpArgs<float> args_;

  kernels::KernelManager kmgr_cwt_;
  kernels::KernelManager kmgr_wvlt_;

  std::vector<OutputDesc> wvlt_out_desc_;
  TensorList<GPUBackend> wvlt_out_;

  mm::uptr<T> gpu_a_ptr;
  TensorListView<StorageGPU, T> gpu_a;

  mm::uptr<T> gpu_b_ptr;
  TensorListView<StorageGPU, T> gpu_b;
};

#define PRINTLINE() std::cout << __LINE__ << std::endl;

template <typename T>
bool CwtImplGPU<T>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto type = type2id<T>::value;
  PRINTLINE();

  kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  ctx.scratchpad = &scratchpad;
  PRINTLINE();

  // TensorListView<StorageGPU, T> a_in;
  /*
  a_in.resize(args_.a.size());
  for (size_t i = 0; i < a_in.size(); i++) {
    TensorShape<> single_a_shape = {1};
    a_in[i].shape = single_a_shape;
  }
  */

  gpu_a_ptr = mm::alloc_raw_unique<T, mm::memory_kind::device>(args_.a.size());
  gpu_a = make_tensor_list_gpu<-1>(gpu_a_ptr.get(), uniform_list_shape(args_.a.size(), {1}));
  CUDA_CALL(cudaMemcpyAsync(gpu_a_ptr.get(), args_.a.data(), args_.a.size() * sizeof(T),
                            cudaMemcpyHostToDevice, ctx.gpu.stream));
  std::cout << args_.a << std::endl;
  std::cout << args_.wavelet_args << std::endl;
  PRINTLINE();
  // TensorListView<StorageGPU, T> b_in;
  // gpu_b.shape = uniform_list_shape(args_.a.size(), {1});
  gpu_b_ptr = mm::alloc_raw_unique<T, mm::memory_kind::device>(args_.a.size());
  gpu_b = make_tensor_list_gpu<-1>(gpu_b_ptr.get(), uniform_list_shape(args_.a.size(), {1}));
  CUDA_CALL(cudaMemcpyAsync(gpu_b_ptr.get(), args_.a.data(), args_.a.size() * sizeof(T),
                            cudaMemcpyHostToDevice, ctx.gpu.stream));
  PRINTLINE();
  /*
  b_in.resize(1);
  TensorShape<> single_b_shape = {1};
  b_in[0].shape = single_b_shape;
  */

  PRINTLINE();
  auto &req_wvlt =
      kmgr_wvlt_.Setup<WvltKernel>(0, ctx, gpu_a, gpu_b, WvltSpan(), args_.wavelet_args);
  PRINTLINE();

  wvlt_out_desc_.resize(1);
  wvlt_out_desc_[0].type = type;
  wvlt_out_desc_[0].shape = req_wvlt.output_shapes[0];
  PRINTLINE();

  /*
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_view = view<const T>(input);
  auto type = type2id<T>::value;
  kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  ctx.scratchpad = &scratchpad;

  PRINTLINE();
  TensorListView<StorageGPU, T> wavelet_values;
  wavelet_values.resize(1);
  PRINTLINE();

  WvltSpan span;
  wavelet_values.shape = GetWaveletOutputShape(args_.a.size(), span);
  PRINTLINE();
  std::cout << args_.a << std::endl;
  TensorListView<StorageGPU, const T> a_values;
  PRINTLINE();
  // auto a_data_gpu = std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, args_.a));
  T *a_data_gpu;
  PRINTLINE();
  std::cout << args_.a.size() * sizeof(T) << std::endl;
  CUDA_CALL(cudaMalloc(&a_data_gpu, args_.a.size() * sizeof(T)));
  PRINTLINE();
  CUDA_CALL(
      cudaMemcpy(a_data_gpu, args_.a.data(), args_.a.size() * sizeof(T),
  cudaMemcpyHostToDevice)); PRINTLINE(); a_values.data[0] = a_data_gpu; PRINTLINE();

  TensorListView<StorageGPU, const T> b_values;
  T b[1];
  b[0] = 0;
  auto *b_data_gpu = std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, b));
  b_values.data[0] = b_data_gpu;
  PRINTLINE();

  // here setup wavelet kernel
  SetupForName<T>(args_.wavelet, kmgr_wvlt_, ctx, wavelet_values, a_values, b_values, span,
                  args_.wavelet_args);
  PRINTLINE();
  CwtArgs _arg2;
  _arg2.a = 10;
  auto &req = kmgr_cwt_.Setup<CwtKernel>(0, ctx, in_view, _arg2);
  output_desc.resize(1);
  output_desc[0].type = type;
  output_desc[0].shape = req.output_shapes[0];  // TODO: list of a x nsamples tensors

  // here setup sum (reduce) kernel

  */
  return true;
}

template <typename T>
TensorListShape<> CwtImplGPU<T>::GetWaveletOutputShape(const size_t &a_size, const WvltSpan &span) {
  int in_size = std::ceil((span.end - span.begin) * span.sampling_rate) + 1;
  TensorListShape<> out_shape(1, 3);
  TensorShape<> tshape;
  for (size_t i = 0; i < a_size; i++) {
    // output tensor will be 3-dimensional of shape:
    // a coeffs x b coeffs x signal samples
    tshape = TensorShape<>({a_size, 1, in_size});
    out_shape.set_tensor_shape(i, tshape);
  }
  return out_shape;
}

template <typename T>
void CwtImplGPU<T>::RunImpl(Workspace &ws) {
  kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
  PRINTLINE();
  kernels::KernelContext ctx;
  PRINTLINE();
  ctx.gpu.stream = ws.stream();
  PRINTLINE();
  ctx.scratchpad = &scratchpad;
  PRINTLINE();
  auto &output = ws.Output<GPUBackend>(0);
  PRINTLINE();
  /*
  auto out_view = view<T>(output);
  PRINTLINE();
  */

  output.Resize(wvlt_out_desc_[0].shape, wvlt_out_desc_[0].type);
  auto out_view = view<T>(output);
  kmgr_wvlt_.Run<WvltKernel>(0, ctx, out_view, gpu_a, gpu_b, WvltSpan());
  PRINTLINE();

  // TensorListView<StorageGPU, T> a_in;
  // a_in.shape = uniform_list_shape(args_.a.size(), {1});
  // kmgr_wvlt_.Run<WvltKernel>(0, ctx, output, gpu_a, gpu_b, WvltSpan());
  /*
  const auto &input = ws.Input<GPUBackend>(0);

  auto in_view = view<const T>(input);
  auto out_view = view<T>(output);


  // here run wavelet kernel

  CwtArgs _arg2;
  _arg2.a = 10;

  kmgr_cwt_.Run<CwtKernel>(0, ctx, out_view, in_view, _arg2);

  // here sum (reduce)
  */
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
