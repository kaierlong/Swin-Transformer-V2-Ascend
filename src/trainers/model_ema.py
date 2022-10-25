#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : this code is refer to
#        1. https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
#        2. https://bbs.huaweicloud.com/blogs/304735
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


_ema_op = C.MultitypeFuncGraph("grad_ema_op")
Assign = P.Assign()
AssignAdd = P.AssignAdd()


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    """Apply grad sum to cumulative gradient."""
    return AssignAdd(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMACell(nn.Cell):
    """EMACell Define"""
    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.ema_weights = weights.clone(prefix="_ema_weights")
        self.ema_decay = Tensor(ema_decay, mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, weights):
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.ema_weights, weights)
        return success
