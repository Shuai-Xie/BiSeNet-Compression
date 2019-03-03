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

image_h = 256
image_w = 320  # /32, int/2

abort_classes = [4, 11, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37]


def val(args, model, dataloader_val):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        for i, (data, label, _, _) in enumerate(dataloader_val):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            # predict
            result, _, _ = model(data)
            predict = result.squeeze()  # squeeze -> one hot 37x480x640 [0,1]
            predict = reverse_one_hot(predict)  # label value img 480x640 [0,36] torch.int64
            predict = np.array(predict).astype('uint8')

            # label -> mask label
            label = label.squeeze()  # torch.Size([1, 480, 640]) -> 480,640
            label = np.array(label).astype('uint8')

            # reduce classes 37 -> 14
            # 0 abort 19 classes to background
            for idx in abort_classes:
                label[np.where(label == idx)] = 0

            # whether use big class
            # ======================
            # # 7 table(counter, desk) 12,14
            # label[np.where(label == 12)] = 7
            # label[np.where(label == 14)] = 7
            # 10 bookshelf(shelves) 15
            label[np.where(label == 15)] = 10
            # 13 blinds(curtain) 16
            label[np.where(label == 16)] = 13
            # total: 19 + 4 + 14 = 37
            # ======================
            # use desk, not use table, counter
            for idx in [7, 12]:
                label[np.where(label == idx)] = 0

            # ordered class or trainning will have ce_loss error
            # 0,1,2,3 no change
            label[np.where(label == 5)] = 4
            label[np.where(label == 6)] = 5
            # label[np.where(label == 7)] = 6
            label[np.where(label == 14)] = 6  # use desk, not table, table harms floor
            label[np.where(label == 8)] = 7
            label[np.where(label == 9)] = 8
            label[np.where(label == 10)] = 9
            label[np.where(label == 13)] = 10
            label[np.where(label == 22)] = 11
            label[np.where(label == 25)] = 12
            label[np.where(label == 29)] = 13
            label[np.where(label == 31)] = 14
            mask = label > 0  # remove 0
            # label_m = label.clone()  # 480x640
            label[mask] -= 1  # 1-37 -> 0-36

            if np.sum(mask) > 0:
                precision = len(np.where(predict[mask] == label[mask])[0]) / np.sum(mask)
                precision_record.append(precision)

        dice = np.mean(precision_record)
        print('precision per pixel for validation: %.3f' % dice)
        return dice


med_frq_old = [
    0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
    0.479574, 0.481544, 0.982534, 1.017466, 0.624581,  # 10
    # 0.479574, 0.781544, 0.982534, 1.017466, 0.624581, change table weight
    2.589096, 0.980794, 0.920340, 0.667984, 1.172291,  # 15
    0.862240, 0.921714, 2.154782, 1.187832, 1.178115,  # 20
    1.848545, 1.428922, 2.849658, 0.771605, 1.656668,  # 25
    4.483506, 2.209922, 1.120280, 2.790182, 0.706519,  # 30
    3.994768, 2.220004, 0.972934, 1.481525, 5.342475,  # 35
    0.750738, 4.040773  # 37
]

# out 14
# med_frq[7_] = (med_frq_old[7] + med_frq_old[12] + med_frq_old[14]) / 3 = 0.710107 # old_7
# med_frq[10_] = (med_frq_old[10] + med_frq_old[15]) / 2 = 0.898436 # old_10
# med_frq[13_] = (med_frq_old[13] + med_frq_old[16]) / 2 = 0.89129 # old_13
# med_frq = [
#     0.382900, 0.452448, 0.637584, 0.585595, 0.479574,  # 1,2,3,5,6
#     0.710107, 0.982534, 1.017466, 0.898436, 0.89129,  # 7_,8,9,10_,13_
#     2.209922, 1.656668, 2.790182, 3.994768  # 22,25,29,31
# ]

# not use big class
med_frq = [
    0.382900, 0.452448, 0.637584, 0.585595, 0.479574,  # 1,2,3,5,6
    0.667984, 0.982534, 1.017466, 0.624581, 0.920340,  # 14,8,9,10,13
    1.428922, 1.656668, 2.790182, 3.994768  # 22,25,29,31
]


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
        for i, (data, label, label_16, label_32) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
                label_16 = label_16.cuda()
                label_32 = label_32.cuda()

            # mid supervision
            result, result_16, result_32 = model(data)
            results = (result, result_16, result_32)
            labels = (label, label_16, label_32)
            # multi scale
            losses = []
            for result, label in zip(results, labels):
                mask = label > 0  # remove 0
                label_m = label.clone()  # 480x640
                label_m[mask] -= 1  # class idx equals to pred class idx
                loss_all = ce_loss(result, label_m.long())
                losses.append(
                    torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
            loss = sum(losses)
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
        if epoch % args.checkpoint_step == 0:
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
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
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
        '--epoch_start_i', '1',
        '--cuda', '0',
        '--batch_size', '5',
        '--num_classes', '14',  # 37->14 less class
        '--context_path', 'resnet18_s',  # less fm
        '--pretrained_model_path', './checkpoints/epoch_0.pth'
    ]
    main(params)
