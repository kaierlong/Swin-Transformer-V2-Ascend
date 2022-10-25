#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : 
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

from mindspore.common import dtype as mstype
from mindspore.common import Parameter, Tensor
from mindspore.train import load_checkpoint, save_checkpoint


def torch_load(pretrained_file):
    model = torch.load(pretrained_file, map_location="cpu")['model']
    prefix = "model."
    model_weights = []
    for key in model.keys():
        # print(key, flush=True)
        if "relative_position_index" in key:
            continue

        if "norm" in key:
            if "weight" in key:
                name = prefix + key.replace(".weight", ".gamma")
            elif "bias" in key:
                name = prefix + key.replace(".bias", ".beta")
            else:
                raise ValueError("Not Supported Norm!")
        elif "cpb_mlp" in key:
            name = prefix + key.replace(".cpb_mlp", ".relative_bias.cpb_mlp")
        elif "relative_coords_table" in key:
            name = prefix + key.replace(".relative_coords_table", ".relative_bias.relative_coords_table")
        elif "qkv" in key:
            name = prefix + key
        # elif "q_bias" in key:
        #     name = prefix + key.replace("q_bias", "q.bias")
        # elif "v_bias" in key:
        #     name = prefix + key.replace("v_bias", "v.bias")
        elif "head" in key:
            name = prefix + key
        else:
            name = prefix + key
        model_weights.append(name)

    keys = sorted(model_weights)
    print("{}".format("\n".join(keys)), flush=True)


def ms_load(pretrained_file):
    param_dict = load_checkpoint(pretrained_file)
    model_weights = []
    other_weights = []
    for key, value in param_dict.copy().items():
        if key.startswith("model"):
            model_weights.append(key)
        else:
            other_weights.append(key)

    keys = sorted(model_weights)
    # print("====== model weights ======\n{}".format("\n".join(model_weights)), flush=True)
    # print("====== other weights ======\n{}".format("\n".join(other_weights)), flush=True)
    print("{}".format("\n".join(keys)), flush=True)


def conv_pth2ckpt(pth_file, ckpt_file, cls_map_file):
    print("====== load pth file: {} ======".format(pth_file), flush=True)

    model = torch.load(pth_file, map_location="cpu")['model']
    prefix = "model."
    model_weights = []

    with open(cls_map_file, "r") as fp:
        lines = fp.readlines()
        cls_index = np.array([int(line.strip()) for line in lines])

    for key in model.keys():
        key_weight_dict = {}
        if "relative_position_index" in key:
            continue
        if "relative_coords_table" in key:
            continue
        if "attn_mask" in key:
            continue

        if "norm" in key:
            if "weight" in key:
                name = prefix + key.replace(".weight", ".gamma")
            elif "bias" in key:
                name = prefix + key.replace(".bias", ".beta")
            else:
                raise ValueError("Not Supported Norm!")
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)
        elif "cpb_mlp" in key:
            name = prefix + key.replace(".cpb_mlp", ".relative_bias.cpb_mlp")
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)
        elif "relative_coords_table" in key:
            name = prefix + key.replace(".relative_coords_table", ".relative_bias.relative_coords_table")
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=False)
        elif "qkv" in key:
            name = prefix + key
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)
        # elif "q_bias" in key:
        #     name = prefix + key.replace("q_bias", "q.bias")
        # elif "v_bias" in key:
        #     name = prefix + key.replace("v_bias", "v.bias")
        elif "head" in key:
            name = prefix + key
            key_weight_dict["name"] = name
            if model[key].shape[0] != 1000:
                weight = model[key].numpy()[cls_index]
            else:
                weight = model[key].numpy()
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(weight, dtype=mstype.float32), requires_grad=True)
        else:
            name = prefix + key
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)

        model_weights.append(key_weight_dict)

    print("====== save ckpt file: {} ======".format(ckpt_file), flush=True)
    save_checkpoint(model_weights, ckpt_file)


def demo():
    # ms_pretrained_file = "/Users/kaierlong/Downloads/swin_v2_model/swinv2_base_patch4_window8_2560-290_2502.ckpt"
    ms_pretrained_file = "/Users/kaierlong/Documents/Codes/OpenI/swin_transformer_v2/swin_v2.ckpt"
    ms_load(pretrained_file=ms_pretrained_file)

    # torch_pretrained_file = "/Users/kaierlong/Downloads/swin_v2_model/swinv2_base_patch4_window12_192_22k.pth"
    # torch_load(pretrained_file=torch_pretrained_file)


def main():
    parser = argparse.ArgumentParser(description="convert pth file to ckpt file.")
    parser.add_argument("--pth_file", type=str, required=True, help="pretrained pth file from torch.")
    parser.add_argument("--ckpt_file", type=str, required=True, help="pretrained ckpt file for mindspore.")
    parser.add_argument("--cls_map_file", type=str, required=True, help="file for image22k mapping to image1k.")

    args = parser.parse_args()

    conv_pth2ckpt(
        pth_file=args.pth_file, ckpt_file=args.ckpt_file, cls_map_file=args.cls_map_file)


if __name__ == "__main__":
    # demo()
    main()
