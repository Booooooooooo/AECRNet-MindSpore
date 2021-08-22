import mindspore
from mindspore import nn, Model, ops
from PIL import Image, ImageOps
from mindspore import Tensor
import os
from mindspore import load_checkpoint, load_param_into_net
from models.model import Dehaze, DehazeWithLossCell
from mindspore.common.initializer import Normal, One

# from utils.utils import metrics

# def rescale_img(img_in, scale):
#     resize_bilinear = ops.ResizeBilinear((img_in.shape[2]*scale, img_in.shape[3]*scale))
#     result = resize_bilinear(img_in) 这个GPU版本的算子R1.3版本才会支持
#     return result

def test(loader_test, net):
    # # l1_loss = nn.L1Loss()
    # net = Dehaze(3, 3)
    # net_with_loss = DehazeWithLossCell(net)
    #
    # #导入参数
    # print(os.path.exists(trainedmodel_dir))
    # if os.path.exists(trainedmodel_dir):
    #     param_dict = load_checkpoint(trainedmodel_dir)
    #     print(f'load model parameters from {trainedmodel_dir}')
    #
    #     # load the parameter into net
    #     load_param_into_net(net, param_dict)

    # model = Model(net)
    # # result = model.eval(test_loader)

    _psnr = nn.PSNR()
    _ssim = nn.SSIM()


    psnr = 0.0
    ssim = 0.0
    bic_psnr = 0.0

    # for item in test_loader.create_dict_iterator():

    # print("loader_test_size:", loader_test.get_dataset_size())
    for iteration, batch in enumerate(loader_test.create_dict_iterator(), 1) :
        print("iteration:", iteration)
        hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
        clear = Tensor(batch["gt"], dtype=mindspore.float32)

        output, _, m4, m5 = net(hazy)
        output = output[:, :, :clear.shape[-2], :clear.shape[-1]]
        psnr += _psnr(output, clear).mean()
        ssim += _ssim(output, clear).mean()

        # sr = net(item["input"]) ##TODO：可能出来尺度会不一样，要处理一下
        # sr = model.predict(item["input"])

        # bilinear = rescale_img(lr, 4)
        # bic_psnr += _psnr(bilinear, hr)

    # print(f"PSNR:{psnr / test_loader.get_dataset_size()}, SSIM:{ssim / test_loader.get_dataset_size()}")
    psnr /= loader_test.get_dataset_size()
    ssim /= loader_test.get_dataset_size()
    return psnr.asnumpy(), ssim.asnumpy()