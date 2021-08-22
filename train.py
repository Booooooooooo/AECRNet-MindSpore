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
        self.weights = ParameterTuple(self.network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = sens

    def construct(self, input, target):
        # ops_print(input)
        weights = self.weights
        # loss, psnr, ssim = self.network(input, target)
        loss = self.network(input, target)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(input, target, sens)
        # return F.depend(loss, self.optimizer(grads)), psnr, ssim
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
    # loader_train = train_dataset.batch(opt.bs, drop_remainder=True)
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    net = Dehaze(3, 3)
    net_with_loss = DehazeWithLossCell(net)
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
    #     losses = ckp['losses']
    #     start_step = ckp['step']
    #     max_ssim = ckp['max_ssim']
    #     max_psnr = ckp['max_psnr']
    #     psnrs = ckp['psnrs']
    #     ssims = ckp['ssims']
    #     print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')
    #     print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')

    train_cell = TrainOneStepCell(net_with_loss, optim)
    net.set_train()

    # TODO:Save model config
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1)
    ckpt_cb = ModelCheckpoint(config=ckpt_config, directory='/home/work/user-job-dir/workspace/trained_models',
                                prefix=opt.model_name)

    cb_params = _InternalCallbackParam()
    cb_params.train_network = net
    cb_params.cur_step_num = 0
    cb_params.batch_num = opt.bs
    cb_params.cur_epoch_num = 0


    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    for epoch in range(0, opt.epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        for iteration, batch in enumerate(loader_train.create_dict_iterator(), 1):
            hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
            clear = Tensor(batch["gt"], dtype=mindspore.float32)

            # print(hazy.shape, clear.shape)

            # loss, psnr, ssim = train_cell(hazy, clear)
            loss, losses = train_cell(hazy, clear)
            # print(loss, type(loss), losses, type(losses))
            epoch_loss += losses

        print("Epoch: [%2d] loss: %.8f"
              % ((epoch), epoch_loss.asnumpy(),))

        if (epoch) % (10) == 0:
            print('===> Saving model')
            cb_params.cur_step_num = epoch + 1
            ckpt_cb.step_end(run_context)

    model_files = os.listdir('/home/work/user-job-dir/workspace/trained_models')
    for name in model_files:
        print(name)
        mox.file.rename(f'/home/work/user-job-dir/workspace/trained_models/{name}', f'obs://test-ddag/output/AECRNet/trained_models/{name}')

    # loss = Loss()
    # loss_net = CustomWithLossCell(net, loss)
    # model = Model(loss_net, loss, optim, metrics=metrics)

    # ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    # ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory='ckpt', config=ckpt_cfg)
    # steps_per_epoch = 1600
    # epoch = 100
    # loss_cb = LossMonitor(steps_per_epoch)
    # time_cb = TimeMonitor(data_size=5000)
    # ckpt_cb = SaveCallback(model, loader_test, 'AECRNet')
    # cb = [time_cb, loss_cb, ckpt_cb]
    # model.train(epoch, loader_train, callbacks=cb, dataset_sink_mode=False)