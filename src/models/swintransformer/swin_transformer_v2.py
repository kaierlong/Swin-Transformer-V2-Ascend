#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : swin transformer v2
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mindspore as ms
import mindspore.common.initializer as weight_init
import mindspore.ops.operations as P

from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy
from mindspore import ops
from mindspore.common import Parameter, Tensor

from src.args import args
from src.models.swintransformer.misc import _ntuple, Identity, DropPath1D

ms_version = ms.__version__
if ms_version >= "1.7.0":
    from mindspore.ops import clip_by_value
else:
    from src.models.swintransformer.clip_ops import clip_by_value

to_2tuple = _ntuple(2)

act_layers = {
    "GELU": nn.GELU,
    "gelu": nn.GELU,
}


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None,
                 act_layer=act_layers[args.nonlinearity],
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = np.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


class WindowPartitionConstruct(nn.Cell):
    """WindowPartitionConstruct Cell"""

    def __init__(self, window_size):
        super(WindowPartitionConstruct, self).__init__()

        self.window_size = window_size

    def construct(self, x):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = P.Reshape()(x, (B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C))
        x = P.Transpose()(x, (0, 1, 3, 2, 4, 5))
        x = P.Reshape()(x, (B * H * W // (self.window_size ** 2), self.window_size, self.window_size, C))

        return x


class WindowReverseConstruct(nn.Cell):
    """WindowReverseConstruct Cell"""

    def construct(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        # TODO
        B = windows.shape[0] // (H * W // window_size // window_size)
        x = ops.Reshape()(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
        x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
        x = ops.Reshape()(x, (B, H, W, -1))
        return x


class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) Cell with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=None):
        super().__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]
        self.dim = dim
        # Wh, Ww
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        self.logit_scale = Parameter(
            Tensor(np.log(10 * np.ones((num_heads, 1, 1))), dtype=mstype.float32), requires_grad=True)
        self.max_scale = Tensor(np.log(1. / 0.01), dtype=mstype.float32)

        self.relative_bias = RelativeBias(
            self.window_size, self.num_heads, self.pretrained_window_size)

        self.qkv = nn.Dense(in_channels=dim, out_channels=dim*3, has_bias=False, activation=None)
        if qkv_bias:
            self.q_bias = Parameter(Tensor(np.zeros(dim), dtype=mstype.float32))
            self.k_bias = Parameter(Tensor(np.zeros(dim), dtype=mstype.float32), requires_grad=False)
            self.v_bias = Parameter(Tensor(np.zeros(dim), dtype=mstype.float32))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True, activation=None)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        # operations
        self.batch_matmul_qk = ops.BatchMatMul(transpose_b=True)
        self.batch_matmul_v = ops.BatchMatMul()
        self.bias_add = ops.Add()
        self.cast = ops.Cast()
        self.concat = ops.Concat()
        self.exp = ops.Exp()
        self.l2_normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)
        self.reshape = ops.Reshape()
        self.split = ops.Split(axis=0, output_num=3)
        self.squeeze = ops.Squeeze(axis=0)
        self.transpose = ops.Transpose()

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)
        if self.q_bias is not None:
            qkv_bias = self.concat([self.q_bias, self.k_bias, self.v_bias])
            qkv = self.bias_add(qkv, qkv_bias)
        qkv = self.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        # qkv = self.split(qkv)
        # [B_, N, num_heads, C // num_heads]
        # the following code not supported by mindspore 1.5.0
        # error as below
        # The 'zeros_like_leaf' operation does not support the type
        # [Slice[kMetaTypeNone : kMetaTypeNone : kMetaTypeNone]]
        # q = qkv[0, :, :, :, :]
        # k = qkv[1, :, :, :, :]
        # v = qkv[2, :, :, :, :]
        q, k, v = self.split(qkv)
        q = self.squeeze(q)
        k = self.squeeze(k)
        v = self.squeeze(v)
        # cosine attention
        q = self.l2_normalize(q)
        k = self.l2_normalize(k)
        # [B_, N, num_heads, num_heads]
        attn = self.batch_matmul_qk(q, k)
        logit_scale = clip_by_value(self.logit_scale, clip_value_max=self.max_scale)
        logit_scale = self.exp(logit_scale)
        attn = attn * logit_scale
        # relative bias
        attn = attn + self.relative_bias()

        # mask is preprocessed in SwinTransformerBlock
        if mask is not None:
            nW = mask.shape[1]
            attn = self.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + mask
            attn = self.reshape(attn, (-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # attn = self.cast(attn, mstype.float32)
        # [B_, N, num_heads, C // num_heads]
        x = self.batch_matmul_v(attn, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class RelativeBias(nn.Cell):
    """RelativeBias Cell"""

    def __init__(self, window_size, num_heads, pretrained_window_size=None):
        super(RelativeBias, self).__init__()
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]

        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.SequentialCell(
            [nn.Dense(2, 512, has_bias=True, activation=None),
             nn.ReLU(),
             nn.Dense(512, num_heads, has_bias=False, activation=None)]
        )

        # get relative_coords_table
        relative_h_size = self.window_size[0] * 2 - 1
        relative_w_size = self.window_size[1] * 2 - 1
        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=np.float32)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=np.float32)
        relative_coords_h = relative_coords_h.reshape(relative_h_size, 1).repeat(
            relative_w_size, 1).reshape(relative_h_size, relative_w_size)
        relative_coords_w = relative_coords_w.reshape(1, relative_w_size).repeat(
            relative_h_size, 1).reshape(relative_w_size, relative_h_size).transpose(1, 0)
        relative_coords_table = np.stack((relative_coords_h, relative_coords_w)).transpose(1, 2, 0)
        # [1, 2*Wh-1, 2*Ww-1, 2]
        relative_coords_table = relative_coords_table.reshape(1, relative_h_size, relative_w_size, 2)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        # normalize to -8, 8
        relative_coords_table *= 8
        relative_coords_table = np.sign(relative_coords_table) * np.log2(
            np.abs(relative_coords_table) + 1.0) / np.log2(8)

        # get pair-wise relative position index for each token inside the window
        h_size = self.window_size[0]
        w_size = self.window_size[1]
        coords_h = np.arange(self.window_size[0]).reshape(h_size, 1).repeat(
            w_size, 1).reshape(h_size, w_size)
        coords_w = np.arange(self.window_size[1]).reshape(1, w_size).repeat(
            h_size, 1).reshape(w_size, h_size).transpose(1, 0)
        # [2, Wh, Ww]
        coords = np.stack((coords_h, coords_w))
        # [2, Wh*Ww]
        coords_flatten = coords.reshape(2, -1)
        # [2, Wh*Ww, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # Wh*Ww, Wh*Ww
        relative_position_index = relative_coords.sum(-1)
        relative_position_index = relative_position_index.reshape(-1)

        # parameters
        self.relative_coords_table = Parameter(Tensor(relative_coords_table), requires_grad=False)
        self.relative_position_index = Parameter(Tensor(relative_position_index), requires_grad=False)

        # onehot
        if args.amp_level in ["O1", "O2", "O3"]:
            self.one_hot = nn.OneHot(
                axis=-1, depth=relative_h_size * relative_w_size, dtype=mstype.float16)
        else:
            self.one_hot = nn.OneHot(
                axis=-1, depth=relative_h_size * relative_w_size, dtype=mstype.float32)

        # operations
        self.unsqueeze = ops.ExpandDims()
        self.matmul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.sigmoid = ops.Sigmoid()
        self.transpose = ops.Transpose()

    def construct(self, axis=0):
        # [1, 2*Wh-1, 2*Ww-1, num_heads]
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        # [(2*Wh-1) * (2*Ww-1), num_heads]
        relative_position_bias_table = self.reshape(relative_position_bias_table, (-1, self.num_heads))
        # [Wh*Ww*Wh*Ww]
        # relative_position_index = self.reshape(self.relative_position_index, (-1))
        relative_position_index = self.one_hot(self.relative_position_index)
        # [Wh*Ww*Wh*Ww, num_heads]
        relative_position_bias = self.matmul(relative_position_index, relative_position_bias_table)
        # [Wh*Ww, Wh*Ww, num_heads]
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = self.sigmoid(relative_position_bias) * 16
        relative_position_bias = self.unsqueeze(relative_position_bias, 0)

        return relative_position_bias


class SwinTransformerBlock(nn.Cell):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=act_layers[args.nonlinearity], norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if isinstance(dim, int):
            dim = (dim,)

        self.norm1 = norm_layer(dim, epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=1e-5)
        mlp_hidden_dim = int((dim[0] if isinstance(dim, tuple) else dim) * mlp_ratio)
        self.mlp = Mlp(in_features=dim[0] if isinstance(dim, tuple) else dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49] ==> [1, 64, 1, 49, 49]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        self.window_partition = WindowPartitionConstruct(self.window_size)
        self.window_reverse = WindowReverseConstruct()

    def construct(self, x):
        """construct function"""
        H, W = self.input_resolution
        B, _, C = x.shape

        shortcut = x
        x = P.Reshape()(x, (B, H, W, C,))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
            # shifted_x = numpy.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        # nW*B, window_size*window_size, C
        x_windows = ops.Reshape()(
            x_windows, (-1, self.window_size * self.window_size, C,))

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = P.Reshape()(attn_windows, (-1, self.window_size, self.window_size, C,))
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
            # x = numpy.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2))  # TODO:Don't stupid
        else:
            x = shifted_x

        x = P.Reshape()(x, (B, H * W, C,))
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2)):
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x):
        x = numpy.roll(x, self.shift_size, self.shift_axis)
        return x


class PatchMerging(nn.Cell):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=2 * dim, has_bias=False)
        self.norm = norm_layer([2 * dim,])
        self.H, self.W = self.input_resolution
        self.H_2, self.W_2 = self.H // 2, self.W // 2
        self.H2W2 = int(self.H * self.W // 4)
        self.dim_mul_4 = int(dim * 4)
        self.H2W2 = int(self.H * self.W // 4)

    def construct(self, x):
        """
        x: B, H*W, C
        """
        B = x.shape[0]
        x = P.Reshape()(x, (B, self.H_2, 2, self.W_2, 2, self.dim))
        x = P.Transpose()(x, (0, 1, 3, 4, 2, 5))
        x = P.Reshape()(x, (B, self.H2W2, self.dim_mul_4))
        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Cell):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Cell, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Cell | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,  # TODO: 这里window_size//2的时候特别慢
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x):
        """construct"""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Cell, optional): Normalization layer. Default: None
    """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True, weight_init="TruncatedNormal")

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x):
        """docstring"""
        B = x.shape[0]
        # FIXME look at relaxing size constraints
        x = ops.Reshape()(self.proj(x), (B, self.embed_dim, -1))  # B Ph*Pw C
        x = ops.Transpose()(x, (0, 2, 1))

        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Cell):
    """ Swin Transformer
        A Pynp impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        image_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Cell): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(Tensor(np.zeros(1, num_patches, embed_dim), dtype=mstype.float32))

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer([self.num_features,], epsilon=1e-5)
        self.avgpool = P.ReduceMean(keep_dims=False)
        self.head = nn.Dense(in_channels=self.num_features,
                             out_channels=num_classes, has_bias=True) if num_classes > 0 else Identity()
        self.init_weights()

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(ops.Transpose()(x, (0, 2, 1)), 2)  # B C 1
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def demo():
    input_img = Tensor(np.random.rand(2, 3, 224, 224), dtype=mstype.float32)
    swin_v2 = SwinTransformerV2()
    out = swin_v2(input_img)
    print("====== out: {} ======".format(out.shape), flush=True)


def main():
    demo()


if __name__ == "__main__":
    main()
