# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from pathlib import Path
from abc import ABC

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, get_worker_info
from torchvision import transforms
from typing import Optional

from .datasets import Images, PairedDataset
from .frontend_dataset import FrontendDataset
from .iterable_patches_dataset import IterablePatchesDataset

def _is_valid_file(filename):
    return any([filename.endswith(ext) for ext in ['.png', '.PNG', '.tif', '.TIF']])


def _read_split(split_path, filename):
    return sorted(list(filter(len, Path(split_path, filename).read_text().split(sep='\n'))))

class BaseGamutNetDataModule(LightningDataModule, ABC):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transform = None
    
    def _create_dataloader(self, dataset, num_workers, pin_memory=False):
        return DataLoader(dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          worker_init_fn=self.worker_init_fn,
                          pin_memory=pin_memory)
    
    def train_dataloader(self):
        return self._create_dataloader(self._create_patches(self.train_dataset),
                                       num_workers=self.hparams.train_num_workers, pin_memory=True)

    def val_dataloader(self):
        return self._create_dataloader(self._create_patches(self.val_dataset),
                                       num_workers=self.hparams.val_num_workers)

    def test_dataloader(self):
        return self._create_dataloader(self._create_patches(self.test_dataset),
                                       num_workers=self.hparams.test_num_workers)

    @staticmethod
    def worker_init_fn(_):
        worker_info = get_worker_info()
        worker_id = worker_info.id
        patch_dataset = worker_info.dataset
        frontend_dataset = patch_dataset.dataset
        num_samples = len(frontend_dataset)
        num_workers = worker_info.num_workers
        assert num_samples >= num_workers, '\'num_samples\' must be greater than or equals to \'num_workers\''
        split_size = num_samples // num_workers
        # using Subset instead of slicing
        indices = list(range(worker_id * split_size, (worker_id + 1) * split_size))
        patch_dataset.dataset = Subset(frontend_dataset, indices)

    
class WideGamutNetDataModule(BaseGamutNetDataModule):
    
    def _create_patches(self, dataset):
        return IterablePatchesDataset(dataset=dataset,
                                      hint_mode=self.hparams.hint_mode,
                                      patch_size=self.hparams.patch_size,
                                      max_patches_per_image=self.hparams.max_patches_per_image,
                                      transform=self.transform, )
    
    # download, split, etc...
    # only called on 1 GPU/TPU in distributed
    def setup(self, stage: Optional[str] = None):
        split_path = self.hparams.split_path

        if 'fit' == stage or stage is None:
            train_input = _read_split(split_path, self.hparams.train_input)
            train_target = _read_split(split_path, self.hparams.train_target)
            train_paired_images = PairedDataset(Images(train_input), Images(train_target))
            self.train_dataset = FrontendDataset(train_paired_images)

            val_input = _read_split(split_path, self.hparams.val_input)
            val_target = _read_split(split_path, self.hparams.val_target)
            val_paired_images = PairedDataset(Images(val_input), Images(val_target))
            self.val_dataset = FrontendDataset(val_paired_images)

        if 'test' == stage or stage is None:
            test_input = _read_split(split_path, self.hparams.test_input)
            test_target = _read_split(split_path, self.hparams.test_target)
            test_paired_images = PairedDataset(Images(test_input), Images(test_target))
            self.test_dataset = FrontendDataset(test_paired_images)

        self.transform = transforms.Compose([transforms.ToTensor()])
        
    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        model_parser = ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument('--split_path', type=str, required=True)
        model_parser.add_argument('--train_input', type=str, default='train-input.txt')
        model_parser.add_argument('--train_target', type=str, default='train-target.txt')
        model_parser.add_argument('--val_input', type=str, default='val-input.txt')
        model_parser.add_argument('--val_target', type=str, default='val-target.txt')
        model_parser.add_argument('--test_input', type=str, default='test-input.txt')
        model_parser.add_argument('--test_target', type=str, default='test-target.txt')

        model_parser.add_argument('--hint_mode', choices=['none', 'o2o_all', 'o2o_rgb', ], default='o2o_all')
        model_parser.add_argument('--patch_size', type=int, nargs=2, default=(32, 32))
        model_parser.add_argument('--max_patches_per_image', type=int, default=32000)

        model_parser.add_argument('--batch_size', type=int, default=32)

        model_parser.add_argument('--train_num_workers', type=int, default=2)
        model_parser.add_argument('--val_num_workers', type=int, default=1)
        model_parser.add_argument('--test_num_workers', type=int, default=1)

        return model_parser