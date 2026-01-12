# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import time
import torch

num_gpus = torch.cuda.device_count()
tensors = [torch.rand(10240, 10240).cuda(i) for i in range(num_gpus)]

while True:
    for tensor in tensors:
        _ = torch.matmul(tensor, tensor)
