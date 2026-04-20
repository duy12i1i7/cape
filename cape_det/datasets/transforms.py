from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ResizeLongestSide:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)

    def __call__(self, image: Image.Image, target: dict):
        width, height = image.size
        scale = min(self.max_size / max(width, height), 1.0 if max(width, height) <= self.max_size else self.max_size / max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.BILINEAR)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                target["boxes"] = boxes * boxes.new_tensor([scale, scale, scale, scale])
        target["size"] = (new_height, new_width)
        target["scale"] = scale
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image: Image.Image, target: dict):
        if random.random() >= self.p:
            return image, target
        width, _ = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target["boxes"] = flipped
        return image, target


class ToTensorNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, target: dict):
        import torch

        arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor(self.mean).view(3, 1, 1)
        std = tensor.new_tensor(self.std).view(3, 1, 1)
        return (tensor - mean) / std, target


def build_transforms(config: dict, train: bool):
    dataset_cfg = config.get("dataset", config)
    transforms: list[Callable] = [ResizeLongestSide(dataset_cfg.get("max_size", dataset_cfg.get("image_size", 1024)))]
    if train and dataset_cfg.get("horizontal_flip", True):
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensorNormalize())
    return Compose(transforms)
