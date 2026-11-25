from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BearingImageDataset(Dataset):
    def __init__(self, directory, image_size=128, normalize=False, return_labels=True, cache_images=True):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.files = sorted(self.directory.glob("*.png"), key=lambda p: self._natural_key(p.name))
        if not self.files:
            raise ValueError(f"No PNG images found in {self.directory}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            *([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] if normalize else []),
        ])
        self.return_labels = bool(return_labels)
        self.image_size = int(image_size)
        self.cache_images = bool(cache_images)
        self.cached_images = None
        if self.cache_images:
            self.cached_images = []
            for image_path in self.files:
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (self.image_size, self.image_size):
                        rgb = rgb.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
                    self.cached_images.append(np.array(rgb, dtype=np.uint8, copy=True))

    @staticmethod
    def _natural_key(value):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]

    @staticmethod
    def _label_from_name(name):
        match = re.search(r"label(\d+)", name, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Image filename must contain labelN: {name}")
        return int(match.group(1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.cached_images is not None:
            tensor = self.transform(Image.fromarray(self.cached_images[index]))
        else:
            with Image.open(self.files[index]) as image:
                tensor = self.transform(image.convert("RGB"))
        if not self.return_labels:
            return tensor
        label = self._label_from_name(self.files[index].name)
        return tensor, torch.tensor(label, dtype=torch.long)


def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_transfer_loaders(data_root, source_condition, target_condition, batch_size=40, image_size=128, num_workers=4, normalize=False, seed=0, cache_images=True):
    root = Path(data_root)
    source_train = BearingImageDataset(root / source_condition / "train", image_size, normalize, cache_images=cache_images)
    source_test = BearingImageDataset(root / source_condition / "test", image_size, normalize, cache_images=cache_images)
    target_adapt = BearingImageDataset(root / target_condition / "val", image_size, normalize, return_labels=False, cache_images=cache_images)
    target_eval = BearingImageDataset(root / target_condition / "val", image_size, normalize, cache_images=cache_images)

    generator_source = torch.Generator().manual_seed(int(seed))
    generator_target = torch.Generator().manual_seed(int(seed) + 1)
    pin_memory = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
    if num_workers > 0:
        common["prefetch_factor"] = 2

    loaders = {
        "source_train": DataLoader(source_train, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator_source, **common),
        "source_test": DataLoader(source_test, batch_size=batch_size, shuffle=False, drop_last=False, **common),
        "target_adapt": DataLoader(target_adapt, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator_target, **common),
        "target_eval": DataLoader(target_eval, batch_size=batch_size, shuffle=False, drop_last=False, **common),
    }
    if len(loaders["source_train"]) == 0:
        raise ValueError("Source training split contains fewer samples than one batch")
    if len(loaders["target_adapt"]) == 0:
        raise ValueError("Target adaptation split contains fewer samples than one batch")
    sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    return loaders, sizes
