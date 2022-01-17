import torch
from torch.utils.data import Dataset

from utils.color import to_single, decode_srgb, srgb_to_prop_cat02, decode_prop


class FrontendDataset(Dataset):
    def __init__(self, paired_images):
        assert isinstance(paired_images, Dataset)
        self.paired_images = paired_images

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # grab a pair of input and target images
            input_img, target_img = self.paired_images[indices]

            # make the input and target image prepared
            prep_input_img = srgb_to_prop_cat02(decode_srgb(to_single(input_img)))
            prep_target_img = decode_prop(to_single(target_img))

            # return the prepared input and target images along with the original input image
            # the input_img will be used for further processing
            return prep_input_img, prep_target_img, input_img
        else:
            raise TypeError(f'{type(self)} indices must be integers, not {type(indices)}')