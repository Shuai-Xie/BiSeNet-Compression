import matplotlib
import skimage
import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imageio
import random

image_h = 480
image_w = 640


class Data(Dataset):
    def __init__(self, image_path, depth_path, label_path, transform=None):
        super().__init__()
        self.transform = transform
        self.label_list = glob.glob(os.path.join(label_path, '*.npy'))

        # data list
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.label_list]
        self.image_list = [os.path.join(image_path, x + '.jpg') for x in self.image_name]
        self.depth_list = [os.path.join(depth_path, x + '.png') for x in self.image_name]

    def __getitem__(self, index):
        image = imageio.imread(self.image_list[index])
        depth = imageio.imread(self.depth_list[index])
        label = np.load(self.label_list[index])

        # create one sample
        sample = {
            'image': image,
            'depth': depth,
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_list)


# scale img to 480,640
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,  # 480,640
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)  # keep ori img value range
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):  # eg: scale = (1.0, 1.4)
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)  # random choose a val in this range
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


# rgb img random hsv
class RandomHSV(object):
    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        sample['image'] = img_new

        return sample


# after random scale, then random crop out (480,640) img
class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


# 0.5 chance flip
class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        # np img -> torch img
        image = image.transpose((2, 0, 1))  # H x W x C -> C X H X W
        depth = np.expand_dims(depth, 0).astype(np.float)

        label_8 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                           order=0, mode='reflect', preserve_range=True)
        label_16 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                            order=0, mode='reflect', preserve_range=True)
        label_32 = skimage.transform.resize(label, (label.shape[0] // 32, label.shape[1] // 32),
                                            order=0, mode='reflect', preserve_range=True)

        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label_8': torch.from_numpy(label_8).float(),
                'label_16': torch.from_numpy(label_16).float(),
                'label_32': torch.from_numpy(label_32).float()}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
        depth = transforms.Normalize(mean=[19050],
                                     std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToRGBD(object):
    def __call__(self, sample):
        image, depth, label, label_8, label_16, label_32 = \
            sample['image'], sample['depth'], sample['label'], sample['label_8'], sample['label_16'], sample['label_32']
        rgbd = torch.cat((image, depth), 0)
        return rgbd, label, label_8, label_16, label_32


if __name__ == '__main__':
    dataset = Data(image_path='/home/disk1/xs/sun/train/image',
                   depth_path='/home/disk1/xs/sun/train/depth',
                   label_path='/home/disk1/xs/sun/train/label_npy',
                   transform=transforms.Compose([scaleNorm(),
                                                 RandomScale((1.0, 1.4)),
                                                 RandomHSV((0.9, 1.1),
                                                           (0.9, 1.1),
                                                           (25, 25)),
                                                 RandomCrop(image_h, image_w),
                                                 RandomFlip(),
                                                 ToTensor(),
                                                 Normalize(),
                                                 ToRGBD()]))
    for i, (rgbd, label, label_8, label_16, label_32) in enumerate(dataset):
        print(rgbd.size())
        print(label.size())
        print(label_8.size())
        print(label_16.size())
        print(label_32.size())
        break
