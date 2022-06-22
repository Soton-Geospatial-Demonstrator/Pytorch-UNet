import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        scale: float = 1.0,
        mask_prefix: str = '',
        mask_suffix: str = '',
        use_n: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_prefix = mask_prefix
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(filepath)[0]
            for filepath in images_dir.rglob(
                "[!.]*[!ini]"
            )
            if filepath.is_file()
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if not use_n == 0:
            assert use_n > 21, f"use_n must be >= 20 (has been set to {use_n})"
            if use_n > len(self.ids):
                logging.info(f"use_n = {use_n} > num_imgs = {len(self.ids)}")
            else:
                self.ids = np.random.choice(
                    self.ids,
                    size=use_n,
                    replace=False,
                )
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        elif not np.isclose(img_ndarray.max(), 0):
            img_ndarray = (~ np.isclose(img_ndarray, 0.0)).astype(int)
            # If mask is not 100% black:
            #   Make sure mask is black-white 1-bit pixel values

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = Path(self.ids[idx]).name
        mask_name = self.mask_prefix + name + self.mask_suffix
        # Check no duplicate files with different file types present
        mask_file = list(self.masks_dir.rglob(mask_name + '.*'))
        img_file = list(self.images_dir.rglob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {mask_name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, use_n=0):
        super().__init__(
            images_dir,
            masks_dir,
            scale,
            mask_suffix='_mask',
            use_n=use_n,
        )
