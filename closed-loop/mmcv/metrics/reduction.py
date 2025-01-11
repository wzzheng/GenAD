# Copyright The PyTorch Lightning team.
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
import torch

from .utils import reduce as __reduce
from .distributed import rank_zero_warn

def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    rank_zero_warn(
        "This `reduce` was deprecated in v1.1.0 in favor of"
        " `mmcv.pytorch_lightning.metrics.utils import reduce`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __reduce(to_reduce=to_reduce, reduction=reduction)

