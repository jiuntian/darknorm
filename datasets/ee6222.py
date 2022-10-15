import logging

import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
from . import img_transforms


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, source):
    # root:'./datasets/hmdb51_frames'
    # source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for ee6222 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])  # 视频名称
                duration = int(line_info[1])  # 视频帧长
                target = int(line_info[2])  # 视频类别
                item = (clip_path, duration, target)
                clips.append(item)
    return clips  # (视频名称,帧长,标签)


def read_segment_rgb_light(path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration, gamma,
                           method):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR  # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE  # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length + 1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            #####
            if method == 'gamma':
                cv_img_origin = img_transforms.gamma_intensity_correction(cv_img_origin, gamma)
            elif method == 'histogram':
                cv_img_origin = img_transforms.histogram_normalization(cv_img_origin)
            elif method == 'gamma_histogram':
                cv_img_origin = img_transforms.gamma_intensity_correction(cv_img_origin, gamma)
                cv_img_origin = img_transforms.histogram_normalization(cv_img_origin)
            elif method == 'none':
                cv_img_origin = cv_img_origin
            else:
                raise NotImplementedError(f'method {method} not defined')
            #####
            if cv_img_origin is None:
                print("Could not load file %s" % (frame_path))
                sys.exit()
                # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class EE6222(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training=False,
                 gamma=None,
                 method='gamma',
                 light=False):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source)
        self.gamma = gamma

        if len(clips) == 0:
            raise (RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                                                                  "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality
        self.light = light

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.ensemble_training = ensemble_training

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.method = method
        logging.info(f"Dataset: Use {'light' if self.light else 'dark'}!")

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        # clips:(视频名称, 帧长, 标签)
        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        # 帧长/分块数
        average_part_length = int(np.floor((duration - self.new_length) / self.num_segments))
        # 取64帧后剩下几帧
        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # offset=2,
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.new_length:
                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id * average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1) / 2 + seg_id * average_duration))
                elif duration >= self.new_length:
                    offsets.append(int((seg_id * average_part_length + (seg_id + 1) * average_part_length) / 2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")

        if self.light:
            clip_input = read_segment_rgb_light(path,
                                                offsets,
                                                self.new_height,
                                                self.new_width,
                                                self.new_length,
                                                self.is_color,
                                                self.name_pattern,
                                                duration,
                                                gamma=self.gamma,
                                                method=self.method
                                                )
        else:
            clip_input = read_segment_rgb_light(path,
                                                offsets,
                                                self.new_height,
                                                self.new_width,
                                                self.new_length,
                                                self.is_color,
                                                self.name_pattern,
                                                duration,
                                                gamma=self.gamma,
                                                method='none'
                                                )

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)
        return clip_input, target

    def __len__(self):
        return len(self.clips)
