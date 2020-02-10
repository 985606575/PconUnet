import random
import torch
from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob("{:s}/data_large/*.jpg".format(img_root),
                              recursive=True)
        else:
            self.paths = glob("{:s}/{:s}_large/*.jpg".format(img_root, split))

        self.mask_paths = glob("{:s}/mask1/*.jpg".format(mask_root))#注意更换mask的时候需要改动
        
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[0])#random.randint(0, self.N_mask - 1+1)
        
        mask = self.mask_transform(mask.convert('RGB'))
        mask=1-mask#注意这个地方在不做目标图检测的时候不不要取反
        
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
