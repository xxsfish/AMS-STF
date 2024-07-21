from pathlib import Path
import imageio
import numpy as np
import math
from enum import Enum, auto, unique
import torch
from torch.utils.data import Dataset
from utils import make_tuple
import torchvision.transforms.functional as TF


@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()


cached_image = {}
cached_path = {}


def get_pair_path(directory: Path, mode: Mode, dataset_name: str):
    if not str(directory) in cached_path:
        prev_label, pred_label, next_label = directory.name.split('-')
        prev_tokens, pred_tokens, next_tokens = prev_label.split('_'), pred_label.split('_'), next_label.split('_')

        # For Tianjin Datasets
        def match_tianjin(path: Path):
            return {
                f'M-{prev_tokens[0]}-{int(prev_tokens[1])}-{int(prev_tokens[2])}' in path.stem: 0,
                f'L-{prev_tokens[0]}-{int(prev_tokens[1])}-{int(prev_tokens[2])}' in path.stem: 1,

                f'M-{next_tokens[0]}-{int(next_tokens[1])}-{int(next_tokens[2])}' in path.stem: 2,
                f'L-{next_tokens[0]}-{int(next_tokens[1])}-{int(next_tokens[2])}' in path.stem: 3,

                f'M-{pred_tokens[0]}-{int(pred_tokens[1])}-{int(pred_tokens[2])}' in path.stem: 4,
                f'L-{pred_tokens[0]}-{int(pred_tokens[1])}-{int(pred_tokens[2])}' in path.stem: 5,
            }

        # For CIA and LGC Datasets
        def match_cia(path: Path):
            return {
                prev_tokens[0] + prev_tokens[1] in path.stem: 0,
                prev_tokens[0] + prev_tokens[2] in path.stem and 'QA' not in path.stem: 1,

                next_tokens[0] + next_tokens[1] in path.stem: 2,
                next_tokens[0] + next_tokens[2] in path.stem and 'QA' not in path.stem: 3,

                pred_tokens[0] + pred_tokens[1] in path.stem: 4,
                pred_tokens[0] + pred_tokens[2] in path.stem and 'QA' not in path.stem: 5,
            }

        if dataset_name == 'tianjin':
            match_fn = match_tianjin
        else:
            match_fn = match_cia

        paths = [None] * 6
        for f in Path(directory).glob('*.tif'):
            try:
                k = match_fn(f)[True]
                paths[k] = f.absolute().resolve()
            except KeyError:
                continue

        cached_path[str(directory)] = paths
    else:
        # cached
        paths = cached_path[str(directory)]

    if mode is Mode.PREDICTION:
        return paths[:-1]  # ignore tag
    else:
        return paths


def load_image_pair(directory: Path, mode: Mode, fast_load=False, padding=[0, 0], scale=1.0, remove_minus=False,
                    dataset_name='', bands=None):
    # Load .tif images with cache
    paths = get_pair_path(directory, mode, dataset_name=dataset_name)
    images = []
    for p in paths:
        if fast_load and (str(p) in cached_image):
            images.append(cached_image[str(p)])
            # print('cached HIT:' + str(p))
        else:
            im = imageio.imread(str(p))  # read image, return (C x H x W)
            im = np.transpose(im, (2, 0, 1))

            # extract bands
            if bands:
                _, h, w = im.shape
                target_matrix = np.zeros((len(bands), h, w))
                for i in range(len(bands)):
                    target_matrix[i, :, :] = im[bands[i] - 1, :, :]
                im = target_matrix

            # Padding
            im = np.pad(im, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'reflect')

            # scale
            if scale != 1.0:
                im = im * scale

            # remove invalid value like -9999
            if remove_minus:
                im = np.maximum(im, 0)
                
            cached_image[str(p)] = im  # cache
            images.append(im)
            # print('cached NOT HIT:' + str(p))

    return images


class PatchSet(Dataset):
    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, padding=0,
                 mode=Mode.TRAINING, x_ranges=None, y_ranges=None, remove_minus=False, fast_load=False,
                 half=False, dataset_name='', dn_max=10000, enable_transform=False, bands=None):
        super(PatchSet, self).__init__()
        self.root_dir = image_dir
        self.image_size = make_tuple(image_size)
        self.x = 0
        self.y = 0
        self.remove_minus = remove_minus
        self.fast_load = fast_load
        self.half = half
        self.dataset_name = dataset_name
        self.dn_max = dn_max
        self.enable_transform = enable_transform
        self.bands = bands

        if x_ranges is not None:
            self.image_size[0] = x_ranges[1] - x_ranges[0]
            self.x = x_ranges[0]

        if y_ranges is not None:
            self.image_size[1] = y_ranges[1] - y_ranges[0]
            self.y = y_ranges[0]

        self.patch_size = make_tuple(patch_size)
        self.patch_stride = self.patch_size if patch_stride is None else make_tuple(patch_stride)
        self.padding = make_tuple(padding)
        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)
        self.mode = mode

        self.x = self.x + self.padding[0]
        self.y = self.y + self.padding[1]

        self.num_patches_x = math.ceil((self.image_size[0] - self.patch_size[0] + 1) / self.patch_stride[0])
        self.num_patches_y = math.ceil((self.image_size[1] - self.patch_size[1] + 1) / self.patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

    def map_index(self, index):
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)

        scale = 1.0 / self.dn_max
        images = load_image_pair(self.image_dirs[id_n], self.mode,
                                 fast_load=self.fast_load, padding=self.padding,
                                 scale=scale, remove_minus=self.remove_minus,
                                 dataset_name=self.dataset_name,
                                 bands=self.bands)

        patches = [None] * len(images)
        for i in range(len(images)):
            im = images[i][:,
                 (self.x + id_x - self.padding[0]): (self.x + id_x + self.patch_size[0] + self.padding[0]),
                 (self.y + id_y - self.padding[1]): (self.y + id_y + self.patch_size[1] + self.padding[1])]

            if self.half:
                patches[i] = torch.from_numpy(im.astype(np.float16))
            else:
                patches[i] = torch.from_numpy(im.astype(np.float32))

        if self.enable_transform:
            # RandomVerticalFlip()
            if torch.rand(1) < 0.5:
                for i in range(len(images)):
                    patches[i] = TF.vflip(patches[i])

            # RandomHorizontalFlip(),
            if torch.rand(1) < 0.5:
                for i in range(len(images)):
                    patches[i] = TF.hflip(patches[i])

        return patches

    def __len__(self):
        return self.num_patches
