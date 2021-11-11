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
import math
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback, SummaryCollector
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
import mindspore.numpy as numpy
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from src.args import args
from src.metric import PSNR
from eval import do_eval
from haze_data import RESIDEDatasetGenerator
# from haze_data import train_de_dataset, eval_ds, device_num, device_id, rank_id
from models.model import Dehaze
from models.edsr_model import EDSR
from src.optimizer import MyOptimizer
from src.contras_loss import ContrastLoss

class NetWithCRLossCell(nn.Cell):
    def __init__(self, net, contrast_w=0, neg_num=0):
        super(NetWithCRLossCell, self).__init__()
        self.net = net
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.contrast_w = contrast_w

    def construct(self, hazy, clear):
        pred = self.net(hazy)

        neg = numpy.flip(hazy, 0)
        neg = neg[:self.neg_num, :, :, :]
        l1_loss = self.l1_loss(pred, clear)
        contras_loss = self.contrast_loss(pred, clear, neg)
        # print(l1_loss, contras_loss)
        loss = l1_loss + self.contrast_w * contras_loss
        return loss

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, hazy, clear):
        weights = self.weights
        loss = self.network(hazy, clear)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(hazy, clear, sens)
        self.optimizer(grads)
        return loss


def train():
    """train"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, global_rank=device_id,
                                          gradients_mean=True)
    if args.modelArts_mode:
        # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        args.dir_data = local_data_url
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=device_id)
        # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)

    train_dataset = RESIDEDatasetGenerator(args, train=True)
    print(len(train_dataset))
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["hazy", "gt"], num_shards=device_num,
                                           # shard_id=rank_id, shuffle=False)
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    eval_dataset = RESIDEDatasetGenerator(args, train=False)
    print(len(eval_dataset))
    eval_ds = ds.GeneratorDataset(eval_dataset, ["hazy", "gt"], shuffle=False)
    eval_ds = eval_ds.batch(1, drop_remainder=True)

    net_m = Dehaze(3, 3, rgb_range=args.rgb_range)
    # net_m = EDSR(args)
    print("Init net weights successfully")

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")
    step_size = train_de_dataset.get_dataset_size()
    lr = []
    for i in range(0, args.epochs):
        cur_lr = 0.5 * (1 + math.cos(i * math.pi / args.epochs)) * args.lr
        # cur_lr = args.lr / (2 ** ((i + 1) // 200))
        lr.extend([cur_lr] * step_size)
    opt = nn.Adam(net_m.trainable_params(), learning_rate=lr, loss_scale=args.loss_scale)
    # opt = nn.Adam(net_m.trainable_params(), learning_rate=lr)
    # opt = MyOptimizer(net_m.trainable_params(), learning_rate=lr, loss_scale=args.loss_scale)
    # opt = MyOptimizer(net_m.trainable_params(), learning_rate=lr,)
    print(net_m.trainable_params())
    # loss_scale_manager = FixedLossScaleManager()
    # loss_scale_manager = DynamicLossScaleManager(init_loss_scale=args.init_loss_scale, \
    #          scale_factor=2, scale_window=1000)

    net_with_loss = NetWithCRLossCell(net_m, args.contra_lambda, args.neg_num)
    train_cell = TrainOneStepCell(net_with_loss, opt)
    net_m.set_train()
    eval_net = net_m

    for epoch in range(0, args.epochs):
        epoch_loss = 0
        for iteration, batch in enumerate(train_de_dataset.create_dict_iterator(), 1):
            hazy = batch["hazy"]
            clear = batch["gt"]

            loss = train_cell(hazy, clear)
            epoch_loss += loss

        print(f"Epoch[{epoch}] loss: {epoch_loss.asnumpy()}")
        # with eval_net.set_train(False):
        #     do_eval(eval_ds, eval_net)

        if (epoch) % 10 == 0:
            print('===> Saving model...')
            save_checkpoint(net_m, f'./ckpt/{args.filename}.ckpt')
            # cb_params.cur_epoch_num = epoch + 1
            # ckpt_cb.step_end(run_context)


if args.modelArts_mode:
        import moxing as mox
        mox.file.copy_parallel(src_url='./log/', dst_url=args.train_url)
        mox.file.copy_parallel(args.ckpt_save_path, args.train_url)

from multiprocessing import Process
from threading import Timer
class RepeatingTimer(Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)
# 子进程要执行的代码
def run_proc():
    if args.modelArts_mode:
        import moxing as mox
        print("Copying...")
        try:
            mox.file.copy_parallel(src_url='./log/', dst_url=args.train_url)
            mox.file.copy_parallel(args.ckpt_save_path, args.train_url)
        except:
            print("No files...")

if __name__ == "__main__":
    time_start = time.time()
    # p = Process(target=run_proc)
    # p.start()
    t = RepeatingTimer(60.0, run_proc)
    t.start()
    train()
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))
