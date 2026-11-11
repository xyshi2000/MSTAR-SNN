"""MSTAR SOC preprocessing adapted from the local ANN/SNN baseline."""
import math
from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


class SquareCrop:
    """Upscale only when needed, then center-crop a fixed square."""

    def __init__(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError('MSTAR input size must be a positive integer')
        self.size = size

    def __call__(self, image):
        if min(image.size) < self.size:
            image = TF.resize(image, self.size, interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True)
        return TF.center_crop(image, [self.size, self.size])


def transforms_for(mean, std, augment=False, input_size=128, resize_if_needed=True):
    crop = SquareCrop(input_size) if resize_if_needed else transforms.CenterCrop(input_size)
    ops = [transforms.Grayscale(1), crop]
    if augment:
        ops.append(transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'))
    return transforms.Compose(ops + [transforms.ToTensor(),
                                     transforms.Normalize([mean], [std])])


def normalization_stats(dataset, input_size=128):
    total = square = 0.0
    pixels = 0
    transform = transforms.Compose([transforms.Grayscale(1), SquareCrop(input_size),
                                    transforms.ToTensor()])
    for path, _ in dataset.samples:
        with Image.open(path) as image:
            x = transform(image).double()
        total += x.sum().item()
        square += x.square().sum().item()
        pixels += x.numel()
    mean = total / pixels
    std = max(max(square / pixels - mean * mean, 0) ** 0.5, 1e-6)
    return mean, std


def load_mstar(root, augment=True, metadata=None, test_only=False, input_size=128):
    """Return datasets and preprocessing; evaluation can use just root/test."""
    root = Path(root)
    train = None if test_only else datasets.ImageFolder(root / 'train')
    test = datasets.ImageFolder(root / 'test')
    if len(test.classes) != 10:
        raise ValueError('MSTAR SOC requires exactly ten classes')
    if train is not None and train.class_to_idx != test.class_to_idx:
        raise ValueError('MSTAR train/test class mappings differ')
    if metadata is None:
        if train is None:
            raise ValueError('MSTAR test-only evaluation requires checkpoint preprocessing')
        resize_if_needed = True
        mean, std = normalization_stats(train, input_size)
        metadata = {'mean': mean, 'std': std, 'classes': train.classes,
                    'class_to_idx': train.class_to_idx,
                    'input_size': [1, input_size, input_size], 'resize_if_needed': True}
    else:
        if (metadata['classes'] != test.classes or
                metadata['class_to_idx'] != test.class_to_idx):
            raise ValueError('MSTAR dataset classes differ from checkpoint')
        saved_size = metadata['input_size']
        if (not isinstance(saved_size, (list, tuple)) or len(saved_size) != 3 or
                any(type(size) is not int or size <= 0 for size in saved_size) or
                saved_size[0] != 1 or saved_size[1] != saved_size[2]):
            raise ValueError('MSTAR checkpoint input size must be [1, size, size]')
        input_size = saved_size[1]
        # Old checkpoints used CenterCrop alone, which pads undersized images.
        resize_if_needed = metadata.get('resize_if_needed', False)
        if not isinstance(resize_if_needed, bool):
            raise ValueError('Invalid MSTAR checkpoint resize_if_needed flag')
        mean, std = metadata['mean'], metadata['std']
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('Invalid MSTAR checkpoint normalization')
    if train is not None:
        train.transform = transforms_for(mean, std, augment, input_size, resize_if_needed)
    test.transform = transforms_for(mean, std, input_size=input_size,
                                    resize_if_needed=resize_if_needed)
    return train, test, metadata
