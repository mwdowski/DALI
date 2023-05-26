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

def get_data():
    s1 = [2.3, 4.5, 1.2, 4.8, 6.8]
    s2 = [5.3, 4.6, 10.2, 0.8, 0.3]

    return np.array([s1, s2], dtype=np.float32)

@pipeline_def(num_threads = 1, device_id = 0)
def get_pipeline():
    data = fn.external_source([get_data()], batch=True, ndim=1)
    result = fn.cwt(data.gpu(), device="gpu", a=2.0)
    return result

pipe = get_pipeline(batch_size=2, num_threads=1, device_id=0)
pipe.build()
output = pipe.run()
print(output.as_cpu())
