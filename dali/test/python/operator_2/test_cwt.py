# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from nvidia.dali import pipeline_def, fn, types

def get_data(sample_info):
    s1 = np.array([2.3, 4.5, 1000.2, 4.8, 6.8], dtype=np.float32)
    s2 = np.array([5.53, 4.6, 10.2, 0.8, 0.3], dtype=np.float32)
    s3 = np.array([5.3, 94.6, 10.2, 0.8, 0.3], dtype=np.float32)
    s4 = np.array([5.23, 4.6, 10.2, 0.85, 0.3], dtype=np.float32)
    s5 = np.array([5.3, 4.6, 103.2, 0.8, 0.36, 4.4], dtype=np.float32)

    l = [s1, s2, s3, s4, s5]

    return l[sample_info.idx_in_batch]

@pipeline_def(num_threads = 1, device_id = 0)
def get_pipeline():
    data = fn.external_source(get_data, batch=False, dtype=types.FLOAT)
    result = fn.cwt(data.gpu(), device="gpu", a=2)
    return data, result

pipe = get_pipeline(batch_size=5, device_id=0)
pipe.build()
d, r = pipe.run()
print(d)
print(r.as_cpu())
