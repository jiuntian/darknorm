import os
import random
import argparse
import time

import numpy as np
import torch

import datasets
import models
import video_transforms
from loss import CELoss
from utils.util import AverageMeter, accuracy

dataset_folder = "datasets"
g = torch.Generator()
width = 170
height = 128
input_size = 112
length = 64
extension = 'img_{0:05d}.jpg'
val_fileName = "test_split%d.txt" % 1
data_dir = os.path.join(dataset_folder, 'EE6222_frames_test')
test_file = os.path.join(dataset_folder, 'settings', 'EE6222', val_fileName)

clip_mean_light = [0.485, 0.456, 0.406] * 1 * length
clip_std_light = [0.229, 0.224, 0.225] * 1 * length
clip_mean = [0.0702773, 0.06571121, 0.06437492] * 1 * length
clip_std = [0.08475896, 0.08116068, 0.07479476] * 1 * length

parser = argparse.ArgumentParser(description='PyTorch DarkNorm Action Recognition')
parser.add_argument('--ckpt', type=str,
                    default='checkpoints/gamma_ce_EE6222_DarkNormr18_split1_triv_norm_reprod_1234')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234, help='dataloader seed')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--worker', type=int, default=8, help='dataloader worker number')


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(model_path, num_categories, device, multiGPUTrain=True, multiGPUTest=False):
    model = models.__dict__['DarkNorm'](num_classes=num_categories, length=length)
    params = torch.load(model_path, map_location=device)

    if multiGPUTest:
        model = torch.nn.DataParallel(model)
        new_dict = {"module." + k: v for k, v in params['state_dict'].items()}
        model.load_state_dict(new_dict)

    elif multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(params['state_dict'])
    model.to(device)
    model.eval()
    return model


def validate(val_loader, model, criterion, epoch):
    losses_classification = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            (inputs, targets) = input
            inputs = inputs.view(-1, length, 3, input_size,
                                 input_size).transpose(1, 2)
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            ce_loss = criterion(output, targets, epoch)

            # measure accuracy and record loss
            if isinstance(output, tuple) and len(output) > 1:
                output = output[0]
            acc1, acc2 = accuracy(output.data, targets, topk=(1, 2))

            losses_classification.update(
                ce_loss.data.item(), output.size(0))

            top1.update(acc1.item(), output.size(0))
            top2.update(acc2.item(), output.size(0))

        print(f'validate * * Prec@1 {top1.avg:.3f} Prec@5 {top2.avg:.3f}'
              f'Classification Loss {losses_classification.avg:.4f}')
    return top1.avg, top2.avg, losses_classification.avg


if __name__ == "__main__":
    args = parser.parse_args()

    ckpt_location = args.ckpt
    seed = args.seed
    device = torch.device(args.gpu)
    criterion = CELoss().to(device)
    start = time.time()

    light = 'gic' in ckpt_location
    dark_std = 'norm' in ckpt_location and not light
    gamma = 1.8 if 'g1.8' in ckpt_location else 3.0

    model_path = os.path.join(ckpt_location, 'model_best.pth.tar')
    assert os.path.exists(model_path), 'model path not exist'

    if dark_std:
        normalize = video_transforms.Normalize(mean=clip_mean,
                                               std=clip_std)
    else:
        normalize = video_transforms.Normalize(mean=clip_mean_light,
                                               std=clip_std_light)
    val_transform = video_transforms.Compose([
        video_transforms.CenterCrop(input_size),
        video_transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.EE6222(root=data_dir, modality="rgb", source=test_file, phase="val",
                                  is_color=True, new_length=length, new_width=width, new_height=height,
                                  video_transform=val_transform, num_segments=1,
                                  gamma=gamma, method='gamma', light=light)
    g.manual_seed(seed)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.worker, pin_memory=True,
                                             worker_init_fn=seed_worker, generator=g)
    print(f"Eval on {ckpt_location}")
    print(f"DarkStd: {dark_std}; Light: {light} " + (f"Gamma: {gamma}" if light else ""))

    spatial_net = build_model(model_path, 10, device)
    acc1, acc2, loss = validate(val_loader, spatial_net, criterion, -1)
    print(f"Result: top1={acc1:.4f} | top2={acc2:.4f} | loss={loss:.4f}")
    print(f"Evaluation complete in {time.time() - start:.1f}s")
