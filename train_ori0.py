from mindspore.train.callback import ModelCheckpoint, Callback, LossMonitor, TimeMonitor, CheckpointConfig, _InternalCallbackParam, RunContext
from mindspore import Model, context, save_checkpoint, ParameterTuple
from mindspore import nn, Tensor
from mindspore import load_checkpoint, load_param_into_net
import mindspore.ops.composite as C
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.train.summary import SummaryRecord
import mindspore
import moxing as mox
# from mindspore.nn.metrics import Metric
# from mindspore.nn import Metrics
# from mindspore.nn.metric import EvaluationBase

from data import train_dataset, test_dataset
from losses.loss import Loss, CustomWithLossCell
from option import opt, log_dir
from models.model import Dehaze, DehazeWithLossCell

import sys
import os
from mindspore.ops.functional import stop_gradient
from test import test

ops_print = ops.Print()
class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval, filename):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.psnr = 0
        self.filename = filename

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        result = self.model.eval(self.ds_eval)
        if result['psnr'] > self.psnr:
            self.psnr = result['psnr']
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=self.filename+'.ckpt')
            print(f"Saving model with PSNR:{self.psnr}")

class PSNRMetrics(nn.Metric):
    def __init__(self):
        super(PSNRMetrics, self).__init__()
        self.eps = sys.float_info.min
        self.psnr_net = nn.PSNR()
        self.clear()

    def clear(self):
        self.psnr = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        psnr = self.psnr_net(y_pred, y)
        self.psnr += psnr
        self._samples_num += 1

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.psnr / self._samples_num

class SSIMMetrics(nn.Metric):
    def __init__(self):
        super(SSIMMetrics, self).__init__()
        self.eps = sys.float_info.min
        self.ssim_net = nn.SSIM()
        self.clear()

    def clear(self):
        self.ssim = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('SSIM need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        ssim = self.ssim_net(y_pred, y)
        self.ssim += ssim
        self._samples_num += 1

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.ssim / self._samples_num

metrics = {
    "psnr": PSNRMetrics(),
    "ssim": SSIMMetrics()
}
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(self.network.trainable_params()) #可训练的参数
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = sens

    def construct(self, input, target):
        weights = self.weights
        loss = self.network(input, target)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(input, target, sens)
        return F.depend(loss, self.optimizer(grads)), loss

def default_train(loader_train, loader_test):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    net = Dehaze(3, 3)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=opt.lr)
    print(os.path.exists(opt.model_dir))
    if opt.resume and os.path.exists(opt.model_dir):
        if opt.pre_model != 'null':
            param_dict = load_checkpoint('./trained_models/' + opt.pre_model)
        else:
            param_dict = load_checkpoint(opt.pre_model)
        print(f'resume from {opt.model_dir}')

        # load the parameter into net
        load_param_into_net(net, param_dict)
        # load the parameter into optimizer
        load_param_into_net(optim, param_dict)
    else:
        print('train from scratch *** ')

    net.set_train()

    loss = Loss()
    loss_net = CustomWithLossCell(net, loss)
    model = Model(loss_net, loss, optim, metrics=metrics)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=opt.model_name, directory='ckpt', config=ckpt_cfg)
    steps_per_epoch = 1600
    epoch = 100
    loss_cb = LossMonitor(steps_per_epoch)
    # time_cb = TimeMonitor(data_size=5000)
    # ckpt_cb = SaveCallback(model, loader_test, 'AECRNet')
    # cb = [time_cb, loss_cb, ckpt_cb]
    model.train(epoch, loader_train, callbacks=loss_cb, dataset_sink_mode=False)

def train(loader_train, loader_test):
    __Loss = Loss()
    __PSNR = nn.PSNR()
    __SSIM = nn.SSIM()
    max_ssim, max_psnr = 0, 0
    ssims, psnrs = [], []

    net = Dehaze(3, 3)
    net_with_loss = DehazeWithLossCell(net) #根据net和losses.loss.Loss()计算损失
    optim = nn.Adam(params=net.trainable_params(), learning_rate=opt.lr)
    train_cell = TrainOneStepCell(net_with_loss, optim)
    net.set_train()

    ckpt_config = CheckpointConfig(save_checkpoint_steps=1)
    ckpt_cb = ModelCheckpoint(config=ckpt_config, directory='/home/work/user-job-dir/workspace/trained_models',prefix=opt.model_name)
    cb_params = _InternalCallbackParam()
    cb_params.train_network = net
    cb_params.cur_step_num = 0
    cb_params.batch_num = opt.bs
    cb_params.cur_epoch_num = 0
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    save_dir = '/home/work/user-job-dir/workspace/list'
    ck_dir = '/home/work/user-job-dir/workspace/checkpoint'
    if os.path.exists(ck_dir) :
        print("dir exists")
    else:
        os.mkdir(ck_dir)
        print('dir created')

    for epoch in range(0, opt.epochs): #
        epoch_loss, te_psnr, te_ssim = 0, 0, 0
        for iteration, batch in enumerate(loader_train.create_dict_iterator(), 1):
            hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
            clear = Tensor(batch["gt"], dtype=mindspore.float32)
            loss, losses = train_cell(hazy, clear)
            epoch_loss += losses
        print("Epoch: [%2d] loss: %.8f"
              % ((epoch), epoch_loss.asnumpy(),))

        for iteration, batch in enumerate(loader_test.create_dict_iterator(), 1):
            hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
            clear = Tensor(batch["gt"], dtype=mindspore.float32)
            hazy = stop_gradient(hazy) #停止计算梯度
            output, _, m4, m5 = net(hazy)
            output = output[:, :, :clear.shape[-2], :clear.shape[-1]]
            te_psnr += __PSNR(output, clear).mean()
            te_ssim += __SSIM(output, clear).mean()

        epoch_psnr = te_psnr/loader_test.get_dataset_size()
        epoch_ssim = te_ssim/loader_test.get_dataset_size()
        ssims.append(epoch_ssim)
        psnrs.append(epoch_psnr)
        print(f"epoch: {epoch}, psnr: {epoch_psnr}, ssim: {epoch_ssim}")
        if epoch_ssim > max_ssim:
            max_ssim = epoch_ssim
        if epoch_psnr > max_psnr:
            max_psnr = epoch_psnr

            # ckpt_name = 'AECRNet_rdin_' + str(epoch + 1) + '.ckpt'
            ckpt_name = f"{opt.model_name}_{str(epoch+1)}.ckpt"
            save_checkpoint(net, '/home/work/user-job-dir/workspace/checkpoint/' + ckpt_name)
            pathh = '/home/work/user-job-dir/workspace/checkpoint/' + ckpt_name
            print(f'save model parameters to {pathh} successfully')
            model_files = os.listdir('/home/work/user-job-dir/workspace/checkpoint')
            for name in model_files :  # 目录下的文件名
                mox.file.rename(f'/home/work/user-job-dir/workspace/checkpoint/{name}',
                                f'obs://test-ddag/output/AECRNet/AECRNet_Mindspore_cyj/checkpoint/{name}')

        save_list(ssims, save_dir, 'ssim_rdin.txt')
        save_list(psnrs, save_dir, 'psnr_rdin.txt')
        if (epoch) % (10) == 0:
            print('===> Saving model')
            cb_params.cur_step_num = epoch + 1
            ckpt_cb.step_end(run_context)

    # model_files = os.listdir('/home/work/user-job-dir/workspace/trained_models')
    # for name in model_files:
    #     mox.file.rename(f'/home/work/user-job-dir/workspace/trained_models/{name}', f'obs://test-ddag/output/AECRNet/trained_models/{name}')
    return psnrs, ssims


def save_list(List, save_dir, save_name):
    if not os.path.exists(save_dir):
    #     print( "dir exists" )
    # else:
        print("not exist")
        os.mkdir(save_dir)
        print('dir created')
    fileOpen = open(save_dir + '/'+ save_name, 'w')
    for im in List:
        fileOpen.write(str(im))
        fileOpen.write(str('\n'))
    fileOpen.close()
    model_files = os.listdir(save_dir)
    for name in model_files :  # 目录下的文件名
        # print("name:", name)
        mox.file.rename(f'{save_dir}/{name}',
                        f'obs://test-ddag/output/AECRNet/AECRNet_Mindspore_cyj/list/{name}')
    print(f'save {save_name} to /test-ddag/output/AECRNet/AECRNet_Mindspore_cyj/list')
    #读取
    # f = open("sample.txt",'r')
    # table = f.read()
    # f.close()