"""MSTAR SOC preprocessing adapted from the local ANN/SNN baseline."""
import math
from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms


def transforms_for(mean, std, augment=False):
    ops = [transforms.Grayscale(1), transforms.CenterCrop(128)]
    if augment:
        ops.append(transforms.RandomCrop(128, padding=4, padding_mode='reflect'))
    return transforms.Compose(ops + [transforms.ToTensor(),
                                     transforms.Normalize([mean], [std])])


def normalization_stats(dataset):
    total = square = 0.0
    pixels = 0
    transform = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(128),
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


def load_mstar(root, augment=True, metadata=None, test_only=False):
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
        mean, std = normalization_stats(train)
        metadata = {'mean': mean, 'std': std, 'classes': train.classes,
                    'class_to_idx': train.class_to_idx, 'input_size': [1, 128, 128]}
    else:
        if (metadata['classes'] != test.classes or
                metadata['class_to_idx'] != test.class_to_idx):
            raise ValueError('MSTAR dataset classes differ from checkpoint')
        if metadata['input_size'] != [1, 128, 128]:
            raise ValueError('Unsupported MSTAR checkpoint input size')
        mean, std = metadata['mean'], metadata['std']
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('Invalid MSTAR checkpoint normalization')
    if train is not None:
        train.transform = transforms_for(mean, std, augment)
    test.transform = transforms_for(mean, std)
    return train, test, metadata
