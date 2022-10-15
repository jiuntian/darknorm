from __future__ import division

from typing import Optional, List, Dict, Tuple

import torch
import random
import numpy as np
import numbers
import cv2
import math
from torch import Tensor
from torchvision.transforms import TrivialAugmentWide, InterpolationMode, functional as F


class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, video_transforms):
        self.video_transforms = video_transforms

    def __call__(self, clips, clips_light):
        # for t in self.video_transforms[:-1]:
        #     clips, clips_light = t(clips, clips_light)
        # clips = self.video_transforms[-1](clips)
        # clips_light = self.video_transforms[-1](clips_light)
        for t in self.video_transforms:
            clips, clips_light = t(clips, clips_light)
        return clips, clips_light


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, clips, clips_light):
        if isinstance(clips, np.ndarray):
            # handle numpy array
            clips = torch.from_numpy(clips.transpose((2, 0, 1)))
            clips_light = torch.from_numpy(clips_light.transpose((2, 0, 1)))
            # backward compatibility
            return clips.float().div(255.0), clips_light.float().div(255.0)
        if isinstance(clips, torch.Tensor):
            return clips.float().div(255.0), clips_light.float().div(255.0)


class NormalizeBothStream(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Here, the input is a clip, not a single image. (multi-channel data)
    The dimension of mean and std depends on parameter: new_length
    If new_length = 1, it falls back to single image case (3 channel)
    """

    def __init__(self, mean, std, mean_light, std_light):
        self.mean = mean
        self.std = std
        self.mean_light = mean_light
        self.std_light = std_light

    def __call__(self, tensor, tensor_light):
        # TODO: make efficient
        torch_mean = torch.tensor([[self.mean]]).view(-1, 1, 1).float()
        torch_std = torch.tensor([[self.std]]).view(-1, 1, 1).float()
        tensor2 = (tensor - torch_mean) / torch_std

        torch_mean_light = torch.tensor([[self.mean_light]]).view(-1, 1, 1).float()
        torch_std_light = torch.tensor([[self.std_light]]).view(-1, 1, 1).float()
        tensor2_light = (tensor_light - torch_mean_light) / torch_std_light
        return tensor2, tensor2_light


class DeNormalizeBothStream(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Here, the input is a clip, not a single image. (multi-channel data)
    The dimension of mean and std depends on parameter: new_length
    If new_length = 1, it falls back to single image case (3 channel)
    """

    def __init__(self, mean, std, mean_light, std_light):
        self.mean = mean
        self.std = std
        self.mean_light = mean_light
        self.std_light = std_light

    def __call__(self, tensor, tensor_light):
        # TODO: make efficient
        torch_mean = torch.tensor([[self.mean]]).view(-1, 1, 1).float()
        torch_std = torch.tensor([[self.std]]).view(-1, 1, 1).float()
        tensor2 = (tensor * torch_std) + torch_mean

        torch_mean_light = torch.tensor([[self.mean_light]]).view(-1, 1, 1).float()
        torch_std_light = torch.tensor([[self.std_light]]).view(-1, 1, 1).float()
        tensor2_light = (tensor_light * torch_std_light) + torch_mean_light
        return tensor2, tensor2_light


class VideoTrivialAugment(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
            self,
            num_magnitude_bins: int = 31,
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Optional[List[float]] = None,
            max_value: int = 255
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.max_value = max_value

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor, img_light) -> Tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        c, h, w = img.shape

        is_color = False
        if c % 3 == 0:
            is_color = True

        # print(f'is color: {is_color}, ${img.shape}')
        img = (img * 255.).type(torch.uint8)
        img_light = (img_light * 255.).type(torch.uint8)
        if is_color:
            num_imgs = int(c / 3)
            scaled_clips = torch.zeros((c, h, w))
            scaled_clips_light = torch.zeros((c, h, w))
            for frame_id in range(num_imgs):
                scaled_clips[frame_id * 3:frame_id * 3 + 3, :, :] \
                    = _apply_op(img[frame_id * 3:frame_id * 3 + 3, :, :] * self.max_value, op_name, magnitude,
                                interpolation=self.interpolation, fill=fill)
                scaled_clips_light[frame_id * 3:frame_id * 3 + 3, :, :] \
                    = _apply_op(img_light[frame_id * 3:frame_id * 3 + 3, :, :] * self.max_value, op_name, magnitude,
                                interpolation=self.interpolation, fill=fill)
            return scaled_clips, scaled_clips_light
        else:
            raise NotImplementedError('not implement yet')

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def _apply_op(
        img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


# 进了
class CenterCrop(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips, clips_light):
        h, w, c = clips.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        is_color = False
        if c % 3 == 0:
            is_color = True

        if is_color:
            num_imgs = int(c / 3)
            scaled_clips = np.zeros((th, tw, c))
            scaled_clips_light = np.zeros((th, tw, c))
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id * 3:frame_id * 3 + 3]
                cur_img_light = clips_light[:, :, frame_id * 3:frame_id * 3 + 3]
                crop_img = cur_img[y1:y1 + th, x1:x1 + tw, :]
                crop_img_light = cur_img_light[y1:y1 + th, x1:x1 + tw, :]
                assert (crop_img.shape == (th, tw, 3))
                scaled_clips[:, :, frame_id * 3:frame_id * 3 + 3] = crop_img
                scaled_clips_light[:, :, frame_id * 3:frame_id * 3 + 3] = crop_img_light
            return scaled_clips, scaled_clips_light
        else:
            num_imgs = int(c / 1)
            scaled_clips = np.zeros((th, tw, c))
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id:frame_id + 1]
                crop_img = cur_img[y1:y1 + th, x1:x1 + tw, :]
                assert (crop_img.shape == (th, tw, 1))
                scaled_clips[:, :, frame_id:frame_id + 1] = crop_img
            return scaled_clips


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """

    def __call__(self, clips, clips_light):
        # clips = np.fliplr(clips)
        # clips = np.ascontiguousarray(clips)
        if random.random() < 0.5:
            clips = np.fliplr(clips)
            clips = np.ascontiguousarray(clips)
            clips_light = np.fliplr(clips_light)
            clips_light = np.ascontiguousarray(clips_light)
        return clips, clips_light


class MultiScaleCrop(object):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao

    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1,
                 interpolation=cv2.INTER_LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))  # upper left
        offsets.append((0, 4 * w_off))  # upper right
        offsets.append((4 * h_off, 0))  # lower left
        offsets.append((4 * h_off, 4 * w_off))  # lower right
        offsets.append((2 * h_off, 2 * w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2 * w_off))  # top center
            offsets.append((4 * h_off, 2 * w_off))  # bottom center
            offsets.append((2 * h_off, 0))  # left center
            offsets.append((2 * h_off, 4 * w_off))  # right center

            offsets.append((1 * h_off, 1 * w_off))  # upper left quarter
            offsets.append((1 * h_off, 3 * w_off))  # upper right quarter
            offsets.append((3 * h_off, 1 * w_off))  # lower left quarter
            offsets.append((3 * h_off, 3 * w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h in range(len(scale_rates)):
            crop_h = int(base_size * scale_rates[h])
            for w in range(len(scale_rates)):
                crop_w = int(base_size * scale_rates[w])
                # append this cropping size into the list
                if (np.absolute(h - w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def __call__(self, clips, clips_light, selectedRegionOutput=False):
        h, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

        crop_size_pairs = self.fillCropSize(h, w)
        size_sel = random.randint(0, len(crop_size_pairs) - 1)
        crop_height = crop_size_pairs[size_sel][0]
        crop_width = crop_size_pairs[size_sel][1]

        if self.fix_crop:
            offsets = self.fillFixOffset(h, w)
            off_sel = random.randint(0, len(offsets) - 1)
            h_off = offsets[off_sel][0]
            w_off = offsets[off_sel][1]
        else:
            h_off = random.randint(0, h - self.height)
            w_off = random.randint(0, w - self.width)

        scaled_clips = np.zeros((self.height, self.width, c))
        scaled_clips_light = np.zeros((self.height, self.width, c))
        if is_color:
            num_imgs = int(c / 3)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id * 3:frame_id * 3 + 3]
                cur_img_light = clips_light[:, :, frame_id * 3:frame_id * 3 + 3]
                crop_img = cur_img[h_off:h_off + crop_height, w_off:w_off + crop_width, :]
                crop_img_light = cur_img_light[h_off:h_off + crop_height, w_off:w_off + crop_width, :]
                scaled_clips[:, :, frame_id * 3:frame_id * 3 + 3] = cv2.resize(crop_img, (self.width, self.height),
                                                                               self.interpolation)
                scaled_clips_light[:, :, frame_id * 3:frame_id * 3 + 3] = cv2.resize(crop_img_light,
                                                                                     (self.width, self.height),
                                                                                     self.interpolation)
            if not selectedRegionOutput:
                return scaled_clips, scaled_clips_light
            else:
                return scaled_clips, scaled_clips_light, off_sel
        else:
            num_imgs = int(c / 1)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id:frame_id + 1]
                crop_img = cur_img[h_off:h_off + crop_height, w_off:w_off + crop_width, :]
                scaled_clips[:, :, frame_id:frame_id + 1] = np.expand_dims(
                    cv2.resize(crop_img, (self.width, self.height), self.interpolation), axis=2)
            if not selectedRegionOutput:
                return scaled_clips
            else:
                return scaled_clips, off_sel
