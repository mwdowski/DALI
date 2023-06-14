// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_NAME_H_
#define DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_NAME_H_

namespace dali {

/**
 * @brief Supported wavelet names
 */
enum DALIWaveletName {
  DALI_HAAR = 0,
  DALI_GAUS = 1,
  DALI_MEXH = 2,
  DALI_MORL = 3,
  DALI_SHAN = 4,
  DALI_FBSP = 5
};

}  // namespace dali

#endif  // DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_NAME_H_
