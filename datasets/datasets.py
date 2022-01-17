from os import scandir
from imageio import imread
from torch.utils.data import Dataset, get_worker_info
from pathlib import Path
import time


def _get_filenames(path, is_valid_file):
    filenames = []
    for entry in scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            if is_valid_file is None:
                filenames.append(entry.name)
            else:
                if is_valid_file(entry.name):
                    filenames.append(entry.name)
    return filenames


class Filenames(Dataset):

    def __init__(self, root_path, filename_list=None, sort=True, is_valid_file=None):
        self.root_path = root_path
        if not isinstance(root_path, Path):
            self.root_path = Path(root_path)
        assert self.root_path.exists(), f'{self.root_path} does not exist'

        self.filename_list = filename_list
        if not isinstance(filename_list, list):
            self.filename_list = _get_filenames(self.root_path, is_valid_file)
        assert len(self.filename_list) > 0, f'{self.root_path} is empty'

        if sort:
            self.filename_list = sorted(self.filename_list)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a filename at a specific index
            filename = self.root_path / self.filename_list[indices]
            return filename
        else:
            raise TypeError(f'{type(self)} indices must be integers, not {type(indices)}')


class Images(Dataset):

    def __init__(self, filenames, loader=imread):
        assert len(filenames) > 0
        self.filenames = filenames
        assert callable(loader)
        self.loader = loader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return an image at a specific index
            filename = self.filenames[indices]
            filename = filename if isinstance(filename, Path) else Path(filename)
            assert filename.is_file(), f"{filename} must be an existing file."
            # To test multi-process loading,
            # 1. import get_worker_info from torch.utils.data
            # 2. uncomment following two lines:
            started_at = time.time()
            image = self.loader(filename)
            worker_info = get_worker_info()
#             print(f'\nWorker {-1 if worker_info is None else worker_info.id}'
#                   f' loaded {filename} in {time.time() - started_at:.2f}s (ind={indices}, len={len(self)}).')
            return image
        else:
            raise TypeError(f'{type(self)} indices must be integers, not {type(indices)}')


class PairedDataset(Dataset):
    """
        See: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a paired data at a specific index
            paired_data = tuple(dataset[indices] for dataset in self.datasets)
            return paired_data
        else:
            raise TypeError(f'{type(self)} indices must be integers, not {type(indices)}')
