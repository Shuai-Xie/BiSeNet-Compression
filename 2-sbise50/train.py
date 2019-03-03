import argparse
from torch.utils.data import DataLoader
from dataset import SUN
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import reverse_one_hot, poly_lr_scheduler
from torchvision import transforms

image_h = 480
image_w = 640


def val(args, model, dataloader_val):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        for i, (data, label) in enumerate(dataloader_val):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            # predict
            result = model(data)
            predict = result.squeeze()  # squeeze -> one hot 37x480x640 [0,1]
            predict = reverse_one_hot(predict)  # label value img 480x640 [0,36] torch.int64
            predict = np.array(predict).astype('uint8')

            # label -> mask label
            label = label.squeeze()  # torch.Size([1, 480, 640]) -> 480,640
            label = np.array(label).astype('uint8')
            mask = label > 0  # remove 0
            # label_m = label.clone()  # 480x640
            label[mask] -= 1  # 1-37 -> 0-36

            if np.sum(mask) > 0:
                precision = len(np.where(predict[mask] == label[mask])[0]) / np.sum(mask)
                precision_record.append(precision)

        dice = np.mean(precision_record)
        print('precision per pixel for validation: %.3f' % dice)
        return dice


med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]  # 37


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter()
    step = 0
    # bce_loss = torch.nn.BCEWithLogitsLoss()  # binary_cross_entropy_with_logits
    ce_loss = torch.nn.CrossEntropyLoss(torch.from_numpy(np.array(med_frq)).float(),  # mulit class
                                        size_average=False, reduce=False).cuda()
    for epoch in range(args.epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            result = model(data)
            mask = label > 0  # remove 0
            label_m = label.clone()  # 480x640
            label_m[mask] -= 1  # class idx equals to pred class idx
            loss_all = ce_loss(result, label_m.long())
            loss = torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
            # total loss
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % loss_train_mean)
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))
        if epoch % args.validation_step == 0:
            dice = val(args, model, dataloader_val)
            writer.add_scalar('precision_val', dice, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/temp_disk/xs/sun', help='path of training data')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default='SUN', help='Dataset you are using.')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used for train')
    parser.add_argument('--learning_rate', default=2e-3, type=float, metavar='LR', help='initial learning rate')  # lr = 0.002
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')  # weight decay = 0.0001
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')  # momentum
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=37, help='num of object classes')  # don't use void
    parser.add_argument('--cuda', type=str, default='1', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints', help='path to save model')
    args = parser.parse_args(params)

    # create dataset and dataloader
    train_image_path = os.path.join(args.data, 'train/image')
    train_depth_path = os.path.join(args.data, 'train/depth')
    train_label_path = os.path.join(args.data, 'train/label_npy')

    # val
    val_image_path = os.path.join(args.data, 'val/image')
    val_depth_path = os.path.join(args.data, 'val/depth')
    val_label_path = os.path.join(args.data, 'val/label_npy')

    dataset_train = SUN.Data(train_image_path, train_depth_path, train_label_path,
                             transform=transforms.Compose([SUN.scaleNorm(),
                                                           SUN.RandomScale((1.0, 1.4)),
                                                           SUN.RandomHSV((0.9, 1.1),
                                                                         (0.9, 1.1),
                                                                         (25, 25)),
                                                           SUN.RandomCrop(image_h, image_w),
                                                           SUN.RandomFlip(),
                                                           SUN.ToTensor(),
                                                           SUN.Normalize(),
                                                           SUN.ToRGBD()]))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_val = SUN.Data(val_image_path, val_depth_path, val_label_path,
                           transform=transforms.Compose([SUN.scaleNorm(),
                                                         SUN.RandomScale((1.0, 1.4)),
                                                         SUN.RandomHSV((0.9, 1.1),
                                                                       (0.9, 1.1),
                                                                       (25, 25)),
                                                         SUN.RandomCrop(image_h, image_w),
                                                         SUN.RandomFlip(),
                                                         SUN.ToTensor(),
                                                         SUN.Normalize(),
                                                         SUN.ToRGBD()]))
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,  # this has to be 1
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path, phase_train=True)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()  # use multiple gpus, batch / gpus, split data

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # ckpt path
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--epoch_start_i', '0',
        '--cuda', '2',
        '--batch_size', '4',
        '--context_path', 'resnet50',
        # '--pretrained_model_path', './checkpoints/epoch_160.pth'
    ]
    main(params)
