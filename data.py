import mindspore.dataset as ds
import matplotlib.pyplot as plt
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.ops as ops
# from mindspore import Tensor

import numpy as np
from PIL import Image
import os
import random

from option import opt

# DATA_DIR = '/home/why/datasets/NH-HAZE'

class DatasetGenerator:
    def __init__(self, path, train, format='.png'):
        # self.size = size
        self.format = format
        self.train = train
        if train:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'train', 'hazy'))
            self.haze_imgs = [os.path.join(path, 'train', 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'train', 'gt')
        else:
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'test', 'hazy'))
            self.haze_imgs = [os.path.join(path, 'test', 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'test', 'gt')
        # print(self.haze_imgs_dir, self.clear_dir)

        np.random.seed(58)
        self.__random_seed = []
        for _ in range(len(self.haze_imgs)):
            self.__random_seed.append(random.randint(0, 1000000))
        self.__index = -1

    def __getitem__(self, index):
        self.__index += 1
        if self.__index >= len(self.haze_imgs):
            self.__index = 0

        haze = Image.open(self.haze_imgs[index]) #读取第index张图片
        img=self.haze_imgs[index].split('/')[-1] #图片的名字
        img_name = img.split('_')
        clear_name=f"{img_name[0]}_gt_{img_name[2]}"
        clear=Image.open(os.path.join(self.clear_dir,clear_name))

        w, h = clear.size
        nw, nh = haze.size
        left = (w - nw)/2
        top = (h - nh)/2
        right = (w + nw)/2
        bottom = (h + nh)/2
        clear = clear.crop((left, top, right, bottom)) #按中心裁剪，使clear和hazy大小一样

        return (haze, clear, index)

    def __len__(self):
        # print("haze images:", len(self.haze_imgs))
        return len(self.haze_imgs)

    def get_seed(self):
        seed = self.__random_seed[self.__index]
        return seed

def decode(img):
    return Image.fromarray(img)

def set_random_seed(img_name, seed):
    random.seed(seed)
    return img_name

ds.config.set_seed(8)
DATA_DIR = opt.data_url

train_dataset_generator = DatasetGenerator(DATA_DIR, train=True)
train_dataset = ds.GeneratorDataset(train_dataset_generator, ["hazy", "gt", "img_name"], shuffle=True)
test_dataset_generator = DatasetGenerator(DATA_DIR, train=False)
test_dataset = ds.GeneratorDataset(test_dataset_generator, ["hazy", "gt", "img_name"], shuffle=False)

transforms_list = [
    decode,
    (lambda img_name: set_random_seed(img_name, train_dataset_generator.get_seed())),
    py_trans.RandomCrop(opt.crop_size),
    py_trans.ToTensor(),
]
compose_trans = Compose(transforms_list)
train_dataset = train_dataset.map(operations=compose_trans, input_columns=["hazy"])
train_dataset = train_dataset.map(operations=compose_trans, input_columns=["gt"])
train_dataset = train_dataset.batch(opt.bs, drop_remainder=True)

# test_dataset也需要裁剪成240*240，batch必须变为4维，否则将在mindspore.conv2D时报错
test_transforms_list = [
    decode,
    (lambda img_name: set_random_seed(img_name, test_dataset_generator.get_seed())),
    py_trans.RandomCrop(opt.crop_size),
    py_trans.ToTensor(),
]
compose_trans = Compose(transforms_list)
test_dataset = test_dataset.map(operations=compose_trans, input_columns=["hazy"])
test_dataset = test_dataset.map(operations=compose_trans, input_columns=["gt"])
test_dataset = test_dataset.batch(5, drop_remainder=False)

if __name__ == '__main__':
    for i in range(2):
        print(i)
        for batch in train_dataset.create_dict_iterator():
            # print(batch)
            # hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
            # clear = Tensor(batch["gt"], dtype=mindspore.float32)

            print(batch["hazy"].shape, batch["gt"].shape)
    # dataset_generator = DatasetGenerator(DATA_DIR, train=True, size=192)
    # dataset = ds.GeneratorDataset(dataset_generator, ["hazy", "gt", "index"], shuffle=False)
    #
    # transforms_list = [
    #     decode,
    #     (lambda img_name: set_random_seed(img_name, dataset_generator.get_seed())),
    #     py_trans.RandomCrop(192),
    #     py_trans.ToTensor(),
    # ]
    # compose_trans = Compose(transforms_list)
    # dataset = dataset.map(operations=compose_trans, input_columns=["hazy"])
    # dataset = dataset.map(operations=compose_trans, input_columns=["gt"])
    #
    # hazy_list, gt_list = [], []
    # for data in dataset.create_dict_iterator():
    #     hazy_list.append(data['hazy'])
    #     gt_list.append(data['gt'])
    #     print("Transformed image Shape:", data['hazy'].shape, ", Transformed label:", data['gt'].shape)
    #
    # num_samples = 5
    # per = ops.Transpose()
    # for i in range(num_samples):
    #     plt.subplot(2, num_samples, i+1)
    #     plt.imshow(per(hazy_list[i], (1, 2, 0)).asnumpy())
    #     plt.subplot(2, num_samples, num_samples + i + 1)
    #     plt.imshow(per(gt_list[i], (1, 2, 0)).asnumpy())
    # plt.show()
