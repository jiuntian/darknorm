#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 1 13:39:00 2021
This repository is based on the repository at https://github.com/artest08/LateTemporalModeling3DCNN.
 We thank the authors for the repository.
This repository is authored by Jiajun Chen
We thank the authors for the repository.
"""
import logging
import os
import time
import argparse
import random
import csv

import numpy as np
import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter

import video_transforms
import models
import datasets
from loss import CELoss
from opt.AdamW import AdamW
from utils.util import save_checkpoint, AverageMeter, accuracy

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch DarkNorm Action Recognition')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to dataset setting files')
parser.add_argument('--dataset', '-d', default='EE6222',
                    choices=["EE6222"],
                    help='dataset: EE6222')
parser.add_argument('--arch', '-a', default='dark_light',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         '(default: dark_light)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--iter-size', default=8, type=int,
                    metavar='I', help='iter size to reduce memory usage (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--print-freq', default=40, type=int,
                    metavar='N', help='print frequency (default: 400)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='eval frequency (default: 1)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments in dataloader (default: 1)')
# parser.add_argument('--resume', default='./dene4', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--continue', dest='continue_training', action='store_true',
                    help='continue training')
parser.add_argument('-g', '--gamma', default=1, type=float,
                    help="the value of gamma")
parser.add_argument('--method', default='gamma', type=str, choices=['gamma', 'histogram', 'gamma_histogram'],
                    help='method of light flow')
parser.add_argument('--loss', default='ce', type=str,
                    help='loss [ce]')
parser.add_argument('--tag', default='', type=str, help='tag')
parser.add_argument('--backbone', default='r18', type=str)
parser.add_argument('--uncorrect-norm', action='store_true',
                    default=False, help='uncorrect norm')
parser.add_argument('--no-trivial', action='store_true',
                    default=False, help='disable TrivialWideAugment')
parser.add_argument('--normalize-first', action='store_true',
                    default=False, help='normalize first before trivial')
parser.add_argument('--light', default=False, action='store_true', help='use light stream')

best_acc1 = 0
best_loss = 30
warmUpEpoch = 5


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    global args, best_acc1, model, writer, best_loss, length, width, height, input_size, scheduler, suffix
    args = parser.parse_args()

    seed = 0  # 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    training_continue = args.continue_training
    suffix = f"net={args.backbone}_light={args.light}_triv={not args.no_trivial}_b={args.batch_size}" \
             f"_norm={not args.uncorrect_norm}_{args.tag}"
    headers = ['epoch', 'top1', 'top2', 'loss']
    if not os.path.exists("logs"):
        os.makedirs("logs")
    with open('logs/train_record_%s.csv' % suffix, 'w', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)
    with open('logs/validate_record_%s.csv' % suffix, 'w', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)

    input_size = 112
    width = 170
    height = 128

    save_location = f"./checkpoints/{args.method}_{args.loss}_{args.dataset}_{args.arch}" \
                    f"{args.backbone}_split{str(args.split)}_{args.tag}"
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    stream_handler = logging.StreamHandler()
    logging.basicConfig(filename=f'{save_location}/log.txt', filemode='a',
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s: %(message)s', '%d-%m-%y %H:%M:%S'))
    logging.getLogger().addHandler(stream_handler)
    logging.info(f'work in {suffix}')
    writer = SummaryWriter(save_location)

    # create model

    if args.evaluate:
        logging.info("Building validation model ... ")
        model = build_model_validate()
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    elif training_continue:
        model, start_epoch, optimizer, best_acc1 = build_model_continue()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logging.info("Continuing with best precision: %.3f and start epoch %d and lr: %f" % (
            best_acc1, start_epoch, lr))
    else:
        logging.info("Building model with ADAMW... ")
        model = build_model()
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
        start_epoch = 0

    logging.info("Model %s is loaded. " % args.arch)

    # define loss function (criterion) and optimizer
    if args.loss == 'ce':
        criterion = CELoss().cuda()
    else:
        raise NotImplementedError('Invalid loss specified')

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)

    logging.info(f"Saving everything to directory {save_location}.")
    dataset = f'./datasets/{args.dataset}_frames'

    cudnn.benchmark = True
    length = 64
    # Data transforming
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean_light = [0.485, 0.456, 0.406] * args.num_seg * length
    clip_std_light = [0.229, 0.224, 0.225] * args.num_seg * length
    clip_mean = [0.0702773, 0.06571121, 0.06437492] * args.num_seg * length
    clip_std = [0.08475896, 0.08116068, 0.07479476] * args.num_seg * length

    if args.uncorrect_norm:
        logging.info('Using uncorrected norm')
        normalize = video_transforms.Normalize(mean=clip_mean_light,
                                               std=clip_std_light)
    else:
        logging.info('Using corrected norm')
        if args.light:
            normalize = video_transforms.Normalize(mean=clip_mean_light,
                                                   std=clip_std_light)
        else:
            normalize = video_transforms.Normalize(mean=clip_mean,
                                                   std=clip_std)

    train_transforms = [
        video_transforms.MultiScaleCrop(
            (input_size, input_size), scale_ratios),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),  # from [0-225] to [0-1]
    ]

    if not args.no_trivial:
        logging.info('using trivial')
        train_transforms.extend([
            video_transforms.VideoTrivialAugment(),  # required input [0-1]
            video_transforms.ToTensor(),
        ])
    else:
        logging.info('not using trivial')

    train_transforms.append(normalize)

    logging.info("Train transforms: ")
    logging.info(train_transforms)

    train_transform = video_transforms.Compose(train_transforms)

    val_transform = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor(),
        normalize,
    ])

    # data loading
    train_setting_file = "train_split%d.txt" % (args.split)
    train_split_file = os.path.join(
        args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_split%d.txt" % (args.split)
    val_split_file = os.path.join(
        args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        logging.info("No split file exists in %s directory. Preprocess the dataset first" % (
            args.settings))
    # load dataset
    train_dataset = datasets.EE6222(root=dataset,
                                    modality="rgb",
                                    source=train_split_file,
                                    phase="train",
                                    is_color=is_color,
                                    new_length=length,
                                    new_width=width,
                                    new_height=height,
                                    video_transform=train_transform,
                                    num_segments=args.num_seg,
                                    gamma=args.gamma,
                                    method=args.method,
                                    light=args.light)

    val_dataset = datasets.EE6222(root=dataset,
                                  modality="rgb",
                                  source=val_split_file,
                                  phase="val",
                                  is_color=is_color,
                                  new_length=length,
                                  new_width=width,
                                  new_height=height,
                                  video_transform=val_transform,
                                  num_segments=args.num_seg,
                                  gamma=args.gamma,
                                  method=args.method,
                                  light=args.light)

    logging.info('{} samples found, {} train data and {} test data.'.format(len(val_dataset) + len(train_dataset),
                                                                            len(train_dataset),
                                                                            len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    logging.info(train_loader)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    if args.evaluate:
        acc1, acc2, lossClassification = validate(
            val_loader, model, criterion, -1)
        return

    for epoch in range(start_epoch, args.epochs):
        if optimizer.param_groups[0]['lr'] <= 1e-7:
            logging.info('Cannot further reduce lr, early stopping')
            break
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            acc1, acc2, lossClassification = validate(
                val_loader, model, criterion, epoch)
            writer.add_scalar('data/top1_validation', acc1, epoch)
            writer.add_scalar('data/top2_validation', acc2, epoch)
            writer.add_scalar(
                'data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(lossClassification)
        # remember best prec@1 and save checkpoint

        is_best = acc1 >= best_acc1
        best_loss = min(lossClassification, best_loss)
        best_acc1 = max(acc1, best_acc1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "checkpoint.pth.tar"
            if is_best:
                logging.info("Saved Best Model")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpoint_name, save_location)

    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_name, save_location)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def build_model():
    # args.arch：dark_light
    model = models.__dict__[args.arch](num_classes=10, length=args.num_seg, backbone=args.backbone)
    logging.info(f'loaded model with backbone {args.backbone}')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    return model


def build_model_validate():
    modelLocation = "./checkpoints/" + args.dataset + \
                    "_" + args.arch + "_split" + str(args.split)
    model_path = os.path.join(modelLocation, 'model_best.pth.tar')
    params = torch.load(model_path)
    logging.info(modelLocation)
    model = models.__dict__[args.arch](num_classes=10, length=args.num_seg, backbone=args.backbone)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()
    return model


def build_model_continue():
    modelLocation = "./checkpoints/" + args.dataset + \
                    "_" + args.arch + "_split" + str(args.split)
    model_path = os.path.join(modelLocation, 'model_best.pth.tar')
    params = torch.load(model_path)
    logging.info(modelLocation)
    model = models.__dict__[args.arch](
        num_classes=10, length=args.num_seg)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])

    startEpoch = params['epoch']
    best_acc = params['best_acc1']
    return model, startEpoch, optimizer, best_acc


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top2 = 0.0
    totalSamplePerIter = 0
    for i, input in enumerate(train_loader):
        (inputs, targets) = input
        inputs = inputs.view(-1, length, 3, input_size,
                             input_size).transpose(1, 2)
        inputs = inputs.cuda()
        output = model(inputs)

        targets = targets.cuda()

        lossClassification = criterion(output, targets, epoch)
        lossClassification = lossClassification / args.iter_size

        if isinstance(output, tuple) and len(output) > 1:
            output = output[0]

        acc1, acc2 = accuracy(output.data, targets, topk=(1, 2))
        acc_mini_batch += acc1.item()
        acc_mini_batch_top2 += acc2.item()

        totalLoss = lossClassification
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter += output.size(0)
        if (i + 1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(
                loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch / args.iter_size, totalSamplePerIter)
            top2.update(acc_mini_batch_top2 /
                        args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top2 = 0.0
            totalSamplePerIter = 0.0

        if (i + 1) % args.print_freq == 0:
            logging.info('[%d] time: %.3f loss: %.4f' %
                         (i, batch_time.avg, lossesClassification.avg))

    logging.info(f'train * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@5 {top2.avg:.3f}'
                 f'Classification Loss {lossesClassification.avg:.4f}')
    with open('train_record_%s.csv' % suffix, 'a', newline='') as f:
        record = csv.writer(f)
        record.writerow([epoch, round(top1.avg, 3), round(
            top2.avg, 3), round(lossesClassification.avg, 4)])
    writer.add_scalar('data/classification_loss_training',
                      lossesClassification.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top2_training', top2.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            (inputs, targets) = input
            inputs = inputs.view(-1, length, 3, input_size,
                                 input_size).transpose(1, 2)
            inputs = inputs.cuda()
            output = model(inputs)

            targets = targets.cuda()

            lossClassification = criterion(output, targets, epoch)

            # measure accuracy and record loss
            if isinstance(output, tuple) and len(output) > 1:
                output = output[0]
            acc1, acc2 = accuracy(output.data, targets, topk=(1, 2))

            lossesClassification.update(
                lossClassification.data.item(), output.size(0))

            top1.update(acc1.item(), output.size(0))
            top2.update(acc2.item(), output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logging.info(f'validate * * Prec@1 {top1.avg:.3f} Prec@5 {top2.avg:.3f}'
                     f'Classification Loss {lossesClassification.avg:.4f}')
        with open('validate_record_%s.csv' % suffix, 'a', newline='') as f:
            record = csv.writer(f)
            record.writerow([epoch, round(top1.avg, 3), round(
                top2.avg, 3), round(lossesClassification.avg, 4)])
    return top1.avg, top2.avg, lossesClassification.avg


if __name__ == '__main__':
    main()
