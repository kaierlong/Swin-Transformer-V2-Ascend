# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""init SwinTransformers"""
from .get_swin import swin_tiny_patch4_window7_224

from .get_swin_v2 import swinv2_tiny_patch4_window8_256
from .get_swin_v2 import swinv2_small_patch4_window8_256
from .get_swin_v2 import swinv2_base_patch4_window8_256
from .get_swin_v2 import swinv2_large_patch4_window16_256
from .get_swin_v2 import swinv2_base_patch4_window12to16_192to256_22kto1k_ft
