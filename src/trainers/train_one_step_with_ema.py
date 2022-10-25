#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : this code refer to
#        1. https://bbs.huaweicloud.com/blogs/304735
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mindspore import nn
from mindspore import ops
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.trainers.model_ema import EMACell

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStepWithEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer, scale_sense=1.0,
                 enable_ema=False, ema_decay=0.9999, clip_global_norm_value=0.):
        super(TrainOneStepWithEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.enable_ema = enable_ema
        self.clip_global_norm_value = clip_global_norm_value
        if self.enable_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)

        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))

        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.clip_global_norm_value > 0.:
                grads = ops.clip_by_global_norm(grads, self.clip_global_norm_value)
                grads = self.grad_reducer(grads)
            loss = F.depend(loss, self.optimizer(grads))
            if self.enable_ema:
                self.ema_model(self.weights)
        else:
            self.print("=============Over Flow, skipping=============")
        return loss
