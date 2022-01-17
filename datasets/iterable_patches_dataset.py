from torch.utils.data import Dataset, IterableDataset, get_worker_info
from .patch_utils import generate_patch
import numpy as np
import itertools
from utils.mask import compute_masks


class IterablePatchesDataset(IterableDataset):

    def __init__(self, dataset, hint_mode, patch_size=(128, 128), max_patches_per_image=1000, transform=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

        self.hint_mode = hint_mode

        assert patch_size[0] > 0 and patch_size[1] > 0
        self.patch_size = patch_size

        assert max_patches_per_image > 0
        self.max_patches_per_image = max_patches_per_image

        self.transform = transform

        worker_info = get_worker_info()
        if worker_info is not None:
            self.rng = np.random.default_rng(worker_info.seed)
        else:
            self.rng = np.random.default_rng(2021)

    def __iter__(self):
        return itertools.chain.from_iterable(map(self.process_data, self.dataset))

    def process_data(self, data):
        prep_input_img, prep_target_img, input_img = data

        o2o_mask, m2o_mask, m_inner = compute_masks(input_img)

        # choose a hint
        hints = {'none': None, 'o2o_all': o2o_mask, 'o2o_rgb': m_inner}  # all the hints here
        hint = hints[self.hint_mode]  # choose a particular hint using hint_mode

        if hint is not None:  # append hint to the input
            prep_hint_img = hint.astype(np.float32)  # type-matching
            prep_input_img = np.dstack((prep_input_img, prep_hint_img))

        patch_generator = generate_patch(m2o_mask, self.patch_size, self.rng)  # generate patch using padded mask
        sliced_patch_generator = itertools.islice(patch_generator, self.max_patches_per_image)  # length-limited

        for patch in sliced_patch_generator:
            _, (top, bottom, left, right) = patch
            input_patch = prep_input_img[top:bottom, left:right, :]
            target_patch = prep_target_img[top:bottom, left:right, :]
            # repeat mask along channel-axis to compute loss conveniently
            m2o_mask_patch = m2o_mask[top:bottom, left:right, None].repeat(3, axis=2)

            if self.transform:  # apply transforms such as ToTensor()
                input_patch = self.transform(input_patch)
                target_patch = self.transform(target_patch)
                m2o_mask_patch = self.transform(m2o_mask_patch)

            yield input_patch, target_patch, m2o_mask_patch  # yield a sample in a batch