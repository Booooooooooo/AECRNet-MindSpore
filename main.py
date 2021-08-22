import moxing as mox
mox.file.shift('os', 'mox')

import mindspore

from data import train_dataset, test_dataset
from option import opt
# from train import train, default_train
from train_ori0 import train, default_train

if __name__ == '__main__':

    psnrs, ssims = train(train_dataset, test_dataset)
    print("psnr:", psnrs)
    print("ssim:", ssims)
    # default_train(train_dataset, test_dataset)