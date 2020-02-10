import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
import torchvision.models as models_baseline 
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='snapshots/default/ckpt/1000000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cpu')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])#修改归一化没用。之前已经试过
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.root,args.root, img_transform, mask_transform, 'val')

#model = models_baseline.__dict__['resnet50']().to(device)
#checkpoint = torch.load(args.snapshot)
#model.load_state_dict(checkpoint['state_dict'])


model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, 'result/PconvUnet11.jpg')
