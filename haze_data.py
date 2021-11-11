import mindspore.dataset as ds
import matplotlib.pyplot as plt
from mindspore import context
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.ops as ops
# from mindspore import Tensor

import numpy as np
from PIL import Image, ImageFile
import os
import random
import imageio
from src.data import common
from src.args import args

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RESIDEDatasetGenerator:
    def __init__(self, args, train, format='.png'):
        # self.size = size
        self.args = args
        path = args.dir_data
        self.format = format
        self.train = train
        if train:
            self.data_name = args.data_train[0]
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'train', 'hazy'))
            self.haze_imgs = [os.path.join(path, 'train', 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'train', 'gt')
        else:
            self.data_name = args.data_test[0]
            self.haze_imgs_dir = os.listdir(os.path.join(path, 'test', 'hazy'))
            self.haze_imgs = [os.path.join(path, 'test', 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, 'test', 'gt')
        #print(len(self.haze_imgs))

    def _get_index(self, idx):
        """get_index"""
        if self.train:
            return idx % len(self.haze_imgs)
        return idx

    def _load_file(self, idx):
        """load_file"""
        idx = self._get_index(idx)
        f_hazy = self.haze_imgs[idx]
        filename = f_hazy.split('/')[-1]
        if self.data_name == 'RESIDE':
            img_id = filename.split('_')[0]
            gt_file = f"{img_id}.png"
        elif self.data_name == 'NHHaze':
            img_name = filename.split('_')
            gt_file = f"{img_name[0]}_GT.png"
            # gt_file=f"{img_name[0]}_gt_{img_name[2]}"
        elif self.data_name == 'Dense':
            img_name = filename.split('_')
            gt_file = f"{img_name[0]}_GT.png"
        else:
            print(f"{self.data_name} Dataset not implemented...")
            return None
        gt = imageio.imread(os.path.join(self.clear_dir,gt_file))
        hazy = imageio.imread(f_hazy)
        
        return hazy, gt, filename

    def get_patch(self, hazy, gt):
        """get_patch"""
        if self.train:
            hazy, gt = common.get_patch(
                hazy, gt,
                patch_size=self.args.patch_size,
                scale=1)
            if not self.args.no_augment:
                hazy, gt = common.augment(hazy, gt)
        else:
            ih, iw = hazy.shape[:2]
            gthr = gt[0:ih, 0:iw]
        return hazy, gt

    def __getitem__(self, idx):
        """get item"""
        hazy, gt, _ = self._load_file(idx)
        pair = self.get_patch(hazy, gt)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1]

    def __len__(self):
        print("haze images:", len(self.haze_imgs))
        return len(self.haze_imgs)



if __name__ == '__main__':
    for i in range(2):
        print(i)
        for batch in test_dataset.create_dict_iterator():
            # print(batch)
            # hazy = Tensor(batch["hazy"], dtype=mindspore.float32)
            # clear = Tensor(batch["gt"], dtype=mindspore.float32)

            print(batch["hazy"].shape, batch["gt"].shape)
    