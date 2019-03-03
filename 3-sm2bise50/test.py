import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/epoch_100.pth',
                    help='The path to the pretrained weights of model')
parser.add_argument('--context_path', type=str, default="resnet50", help='The context path model you are using.')
parser.add_argument('--num_classes', type=int, default=37, help='num of object classes (with void)')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped/resized input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
parser.add_argument('--csv_path', type=str, default='/home/disk1/xs/sun/seg37_class_dict.csv', help='Path to label info csv file')

args = parser.parse_args()

# read csv label path
label_info = get_label_info(args.csv_path)
del label_info['background']

scale = (args.crop_height, args.crop_width)

# build model
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # set cuda id
model = BiSeNet(args.num_classes, args.context_path)

# load pretrained model if exists
print('load model from %s ...' % args.checkpoint_path)

# args.use_gpu = False
if torch.cuda.is_available() and args.use_gpu:
    model = torch.nn.DataParallel(model).cuda()  # line 34 has set cuda id
    model.module.load_state_dict(torch.load(args.checkpoint_path))  # GPU -> GPU
else:
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))  # GPU -> CPU

print('Done!')

resize_img = transforms.Resize(scale, Image.BILINEAR)
resize_depth = transforms.Resize(scale, Image.NEAREST)
to_tensor = transforms.ToTensor()

normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
normalize_depth = transforms.Normalize(mean=[19050],
                                       std=[9650])


def predict_RGBD(image, depth):  # nd convenient both for img and video
    # pre-processing on image
    image = resize_img(image)
    image = to_tensor(image).float()  # size:(3,480,640), val:[0,1]
    image = normalize_img(image)

    depth = resize_depth(depth)
    depth = np.expand_dims(depth, 0).astype(np.float)
    depth = torch.from_numpy(depth * 10).float()  # todo: kinect v1 danwei
    depth = normalize_depth(depth)

    rgbd = torch.cat((image, depth), 0)
    rgbd = rgbd.unsqueeze(0)

    # predict
    model.eval()
    predict = model(rgbd).squeeze()
    # predict = reverse_one_hot(predict)
    # predict = colour_code_segmentation(np.array(predict), label_info)
    # predict = np.uint8(predict)
    #
    # return cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR)


def predict_img_dir(img_dir, depth_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    t1 = datetime.now()
    print(t1)
    for img in os.listdir(depth_dir):
        image = Image.open(img_dir + img.replace('.png', '.jpg'))
        depth = Image.open(depth_dir + img)
        predict_RGBD(image, depth)
        # seg = predict_RGBD(image, depth)
        # cv2.imwrite(out_dir + img, seg)
        print(img)
    t2 = datetime.now()
    print(t2)
    # 1s = 1000ms, 1ms = 1000us
    total_ms = (t2 - t1).seconds * 1000 + (t2 - t1).microseconds / 1000
    if t1.microsecond > t2.microsecond:
        total_ms -= 1000
    print(total_ms)
    print('per', total_ms / 40)


def test_sun():
    predict_img_dir(img_dir='./img/sun/rgb/',
                    depth_dir='./img/sun/depth/',
                    out_dir='./img/sun/seg/')


def test_lab():
    predict_img_dir(img_dir='./img/lab/rgb/',
                    depth_dir='./img/lab/depth/',
                    out_dir='./img/lab/seg/')


if __name__ == '__main__':
    test_lab()
