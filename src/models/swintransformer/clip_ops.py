#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : Operations for clipping tensors to min/max values.
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr


@constexpr
def _check_output_shape(input_shape, out_shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if input_shape != out_shape:
        raise ValueError(f"{msg_prefix} input 'x' shape should be equal to the output shape, but got "
                         f"input 'x' shape {input_shape}, output shape {out_shape}.")


def check_np_type(np_dtype, is_max_val):
    if not (np.issubsctype(np_dtype, np.floating) or np.issubsctype(np_dtype, np.integer) or
            np.issubsctype(np_dtype, np.complex64) or np.issubsctype(np_dtype, np.complex128) or
            np.issubsctype(np_dtype, np.bool_)):
        value_info = ("clip_value_max", "clip_value_min") if is_max_val else ("clip_value_min", "clip_value_max")
        raise ValueError(f"When {value_info[0]} is none, The date type of {value_info[1]} only support integer,"
                         f"floating, bool, complex64 or complex128. But got {np_dtype}")


@constexpr
def create_max_min_value(ms_type, is_max_val):
    """create max or min value"""
    np_dtype = mstype.dtype_to_nptype(ms_type)
    check_np_type(np_dtype, is_max_val)
    if np.issubsctype(np_dtype, np.floating):
        val = np.finfo(np_dtype).max if is_max_val else np.finfo(np_dtype).min
    elif np.issubsctype(np_dtype, np.integer):
        val = np.iinfo(np_dtype).max if is_max_val else np.iinfo(np_dtype).min
    elif np.issubsctype(np_dtype, np.complex64):
        val = np.finfo(np.float32).max if is_max_val else np.finfo(np.float32).min
        val = np.complex64(np.complex(val, val))
    elif np.issubsctype(np_dtype, np.complex128):
        val = np.finfo(np.float64).max if is_max_val else np.finfo(np.float64).min
        val = np.complex128(np.complex(val, val))
    else:
        val = np.bool_(True) if is_max_val else np.bool_(False)
    return Tensor(val, ms_type)


@constexpr
def raise_value_error():
    raise ValueError("At least one of 'clip_value_min' or 'clip_value_max' must not be None")


def clip_by_value(x, clip_value_min=None, clip_value_max=None):
    r"""
    Clips tensor values to a specified min and max.

    Limits the value of :math:`x` to a range, whose lower limit is `clip_value_min`
    and upper limit is `clip_value_max` .

    .. math::

        out_i= \left\{
        \begin{array}{align}
            clip\_value\_max & \text{ if } x_i\ge  clip\_value\_max \\
            x_i & \text{ if } clip\_value\_min \lt x_i \lt clip\_value\_max \\
            clip\_value\_min & \text{ if } x_i \le clip\_value\_min \\
        \end{array}\right.

    Note:
        `clip_value_min` needs to be less than or equal to `clip_value_max` . The data type of x, `clip_value_min` and
        `clip_value_max` should support implicit type conversion and cannot all be bool type.

    Args:
          x (Tensor): Input data. The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          clip_value_min (Tensor): The minimum value. `clip_value_min` and `clip_value_max` cannot be all None.
                                   Default: None.
          clip_value_max (Tensor): The maximum value. `clip_value_min` and `clip_value_max` cannot be all None.
                                   Default: None.

    Returns:
          Tensor, a clipped Tensor. The data type is the one with higher precision or higher digits among
          the x, `clip_value_min` and `clip_value_max` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> min_value = Tensor(5, mindspore.float32)
        >>> max_value = Tensor(20, mindspore.float32)
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clip_by_value(x, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
    """

    min_op = P.Minimum()
    max_op = P.Maximum()
    if clip_value_min is None and clip_value_max is None:
        raise_value_error()
    if clip_value_min is None:
        clip_value_min = create_max_min_value(F.dtype(clip_value_max), False)
    if clip_value_max is None:
        clip_value_max = create_max_min_value(F.dtype(clip_value_min), True)
    x_min = min_op(x, clip_value_max)
    x_max = max_op(x_min, clip_value_min)
    _check_output_shape(F.shape(x), F.shape(x_max), 'clip_by_value')
    return x_max
