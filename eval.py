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
"""eval script"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.args import args
from src.metrics import calc_psnr, quantize, calc_ssim
from src.metric import saveSrHr
from models.model import Dehaze
from haze_data import RESIDEDatasetGenerator
# from haze_data import eval_ds

# device_id = int(os.getenv('DEVICE_ID', '0'))
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id, save_graphs=False)
# context.set_context(max_call_depth=10000)
def eval_net():
    # plt.figure()

    """eval"""
    eval_loader = eval_ds.create_dict_iterator(output_numpy=True)
    net_m = Dehaze(3, 3)
    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)
    net_m.set_train(False)

    print('load mindspore net successfully.')
    num_imgs = eval_ds.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(eval_loader):
        hazy = imgs['hazy']
        gt = imgs['gt']
        hazy = Tensor(hazy, mstype.float32)
        pred = net_m(hazy)
        saveSrHr('log', str(batch_idx), pred, gt)
        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, gt, 1, 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        gt = gt.reshape(gt.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, gt, 1)
        print("current psnr: ", psnr)
        print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s is %.4f' % (args.data_test[0], psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s is %.4f' % (args.data_test[0], ssims.mean(axis=0)[0]))

def do_eval(eval_ds, eval_net):
    train_loader = eval_ds.create_dict_iterator(output_numpy=True)
    num_imgs = eval_ds.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        hazy = imgs['hazy']
        gt = imgs['gt']
        hazy = Tensor(hazy, mstype.float32)
        pred = eval_net(hazy)
        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, gt, 1, 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        gt = gt.reshape(gt.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, gt, 1)
        # print("current psnr: ", psnr)
        # print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s is %.4f' % (args.data_test[0], psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s is %.4f' % (args.data_test[0], ssims.mean(axis=0)[0]))
    return np.mean(psnrs)

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id, save_graphs=False)

    if args.modelArts_mode:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        args.dir_data = local_data_url
    else:
        # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=device_id)
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)

    eval_dataset = RESIDEDatasetGenerator(args, train=False)
    print(len(eval_dataset))
    eval_ds = ds.GeneratorDataset(eval_dataset, ["hazy", "gt"], shuffle=False)
    eval_ds = eval_ds.batch(1, drop_remainder=True)

    context.set_context(max_call_depth=10000)
    time_start = time.time()
    print("Start eval function!")
    eval_net()
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))
