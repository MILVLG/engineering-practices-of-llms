# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
from datetime import datetime


def _debug(msg: str) -> None:
    """Print debug info when MINDSPEED_DEBUG is enabled."""
    if os.environ.get('MINDSPEED_DEBUG'):
        rank = os.environ.get('RANK', 'N/A')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[DEBUG {timestamp}][rank={rank}] modellink.patchs: {msg}', flush=True)


from .megatron_mock import mock_megatron_dependencies, patch_npu_apex_torch
_debug('calling mock_megatron_dependencies()')
mock_megatron_dependencies()
_debug('calling patch_npu_apex_torch()')
patch_npu_apex_torch()

from .megatron_patch import exec_adaptation
_debug('calling exec_adaptation()')
exec_adaptation()
_debug('finished modellink.patchs initialization')
