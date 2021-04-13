import moxing as mox
mox.file.shift('os', 'mox')
import mindspore

from data import train_dataset, test_dataset
from option import opt
from train import train

if __name__ == '__main__':

    train(train_dataset, test_dataset)