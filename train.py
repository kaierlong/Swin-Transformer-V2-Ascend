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
"""train"""
import os
import time

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer


def init_openi(mode="train"):
    if mode == "train":
        work_dir = "/home/work/user-job-dir"
    else:
        work_dir = "/home/ma-user/work/"

    model_dir = os.path.join(work_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    data_dir = os.path.join(work_dir, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    return model_dir, data_dir


def get_lock_file(file_name, mode="train"):
    if mode == "train":
        work_dir = "/home/work/user-job-dir"
    else:
        work_dir = "/home/ma-user/work/"

    file_path = os.path.join(work_dir, file_name)

    return file_path

def get_git_pretrained_ckpt_file(args):
    current_path = os.path.abspath(__file__)
    code_dir = os.path.dirname(current_path)
    ckpt_file_path = os.path.join(code_dir, "src/configs/{}.ckpt".format(args.arch))
    if not os.path.exists(ckpt_file_path):
        raise ValueError("pretrained ckpt file: {} not exists!".format(ckpt_file_path))
    
    return ckpt_file_path

def get_obs_pretrained_ckpt_file(model_dir):
    pretrained_dir = os.path.join(model_dir, "pretrained")
    if not os.path.exists(pretrained_dir):
        os.mkdir(pretrained_dir)
    pretrained_ckpt_file = os.path.join(pretrained_dir, "swinv2_large_patch4_window16_2560-120_3336.ckpt")

    return pretrained_ckpt_file


def sync_data(args, data_lock_file, data_dir):
    obs_data_url = args.data_url
    tag = True
    try:
        import moxing as mox
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir), flush=True)
    except Exception as e:
        print("moxing download {} to {} failed, error:\n{}".format(obs_data_url, data_dir, e), flush=True)
        tag = False

    if tag:
        with open(data_lock_file, "w") as fp:
            fp.write("1\n")


def sync_best_model(args, best_model_dir, rank):
    obs_train_url = args.train_url
    try:
        import moxing as mox
        mox.file.copy_parallel(best_model_dir, os.path.join(obs_train_url, "ckpt_best_{:04d}".format(rank)))
        print("Successfully Upload {} to {}".format(best_model_dir, obs_train_url), flush=True)
    except Exception as e:
        print('moxing upload {} to {} failed, error: \n{}'.format(best_model_dir, obs_train_url, e), flush=True)


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    model_dir = "./model"
    data_dir = ""
    if args.run_openi:
        model_dir, data_dir = init_openi()

    train_dir = os.path.join(model_dir, "ckpt_{:04d}".format(rank))
    best_model_dir = os.path.join(model_dir, "ckpt_best_{:04d}".format(rank))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(best_model_dir):
        os.mkdir(best_model_dir)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained_ckpt_file = get_git_pretrained_ckpt_file(args)
        args.pretrained = pretrained_ckpt_file
        pretrained(args, net)

    if args.run_openi:
        data_lock_file = get_lock_file(file_name="data_lock.txt")
        if rank == 0:
            sync_data(args, data_lock_file, data_dir=data_dir)
        else:
            while True:
                if os.path.exists(data_lock_file):
                    break
                time.sleep(10)
        args.data_url = data_dir
        print("====== all rank data file sync over ======", flush=True)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num,
                                 keep_checkpoint_max=args.best_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())

    model_prefix = "{}_{:04d}".format(args.arch, rank)
    ckpoint_cb = ModelCheckpoint(prefix=model_prefix, directory=train_dir, config=config_ck)

    print_steps = batch_num // 10
    loss_cb = LossMonitor(per_print_times=print_steps)
    eval_cb = EvaluateCallBack(
        model, eval_dataset=data.val_dataset, src_url=train_dir,
        train_url=os.path.join(args.train_url, "ckpt_{:04d}".format(rank)),
        rank=rank, model_prefix=model_prefix, batch_num=batch_num,
        best_model_dir=best_model_dir, best_freq=args.best_every, save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=args.dataset_sink_mode)
    print("train success")

    if args.run_openi:
        sync_best_model(args, best_model_dir, rank=rank)


if __name__ == '__main__':
    main()
