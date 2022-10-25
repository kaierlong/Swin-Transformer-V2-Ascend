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
"""callback function"""

import os
import shutil

from mindspore.train.callback import Callback

from src.args import args


def keep_best_latest_models(best_model_dir, num_keep=5):
    best_model_list = os.listdir(best_model_dir)
    if not best_model_list:
        return []
    else:
        best_model_list = sorted(best_model_list, key=lambda x: os.path.getmtime(os.path.join(best_model_dir, x)),
                                 reverse=True)
        print("====== model list before delete ======\n{}".format("\n".join(best_model_list)), flush=True)

        for model_file in best_model_list[num_keep:]:
            model_path = os.path.join(best_model_dir, model_file)
            if os.path.isfile(model_path):
                os.remove(model_path)
                best_model_list.remove(model_file)
        print("====== model list after delete ======\n{}".format("\n".join(best_model_list)), flush=True)

        return best_model_list


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, rank, model_prefix, batch_num,
                 best_model_dir, best_freq=5, save_freq=50):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.rank = rank
        self.model_prefix = model_prefix
        self.batch_num = batch_num
        self.best_model_dir = best_model_dir
        self.best_freq = best_freq
        self.save_freq = save_freq
        self.best_epoch_acc = [(0, 0.)]

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        result = self.model.eval(self.eval_dataset)
        best_acc = self.best_epoch_acc[-1][1]
        if result["acc"] > best_acc:
            self.best_epoch_acc.append((cur_epoch_num, result["acc"]))
        print("epoch: {:04d} device: {:04d} acc: {}, best epoch: {:04d}, acc is {}".format(
            cb_params.cur_epoch_num, self.rank, result["acc"],
            self.best_epoch_acc[-1][0], self.best_epoch_acc[-1][1]), flush=True)

        if args.run_openi:
            if cur_epoch_num % self.best_freq == 0:
                if len(self.best_epoch_acc) > 5:
                    move_ckpt_list = self.best_epoch_acc[-5:]
                else:
                    move_ckpt_list = self.best_epoch_acc[1:]
                print("====== device: {} move ckpt list ======\n{}".format(self.rank, move_ckpt_list), flush=True)

                for src_epoch_num, _ in move_ckpt_list:
                    ckpt_name = "{}-{}_{}.ckpt".format(self.model_prefix, src_epoch_num, self.batch_num)
                    src_file = os.path.join(self.src_url, ckpt_name)
                    tgt_file = os.path.join(self.best_model_dir, ckpt_name)
                    if not os.path.exists(src_file):
                        print("source model file: {} not exists!".format(src_file), flush=True)
                        continue
                    if not os.path.exists(tgt_file):
                        shutil.copy(src_file, tgt_file)
                        print("target model file: {} copyed!".format(tgt_file), flush=True)
                keep_best_latest_models(self.best_model_dir, num_keep=5)

            import moxing as mox
            if cur_epoch_num % self.save_freq == 0:
                mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)