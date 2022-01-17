import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageCms
from imageio import imread, imsave
from torchvision.transforms.functional import to_tensor

from models import WideGamutNetPL
from utils.color import to_single, to_uint8, decode_srgb, srgb_to_prop_cat02, encode_prop
from utils.mask import compute_masks

# ICC_PROFILE_BYTES = ImageCms.getOpenProfile('icc_profiles/ProPhoto.icm').tobytes()


class NoPWInferenceAgent:

    def __init__(self,
                 version_path,
                 ckpt_filename,
                 ckpt_dirname='checkpoints',
                 hparams_filename='hparams.yaml',
                 device='cpu'):

        self.version_path = Path(version_path)
        assert self.version_path.is_dir(), 'version_path must be an existing directory'

        self.checkpoint_path = self.version_path / ckpt_dirname / ckpt_filename
        assert self.checkpoint_path.is_file(), 'checkpoint_path must be an existing file'

        self.hparams_file = self.version_path / hparams_filename
        assert self.hparams_file.is_file(), 'hparams_file must be an existing file'

        if torch.cuda.is_available() and device != 'cpu':
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        self.model = WideGamutNetPL.load_from_checkpoint(checkpoint_path=str(self.checkpoint_path),
                                                         hparams_file=str(self.hparams_file),
                                                         map_location=map_location)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def single_image_inference(self, img_path, output_img_path=None, icc_profile=None):
        started_at = time.time()
        input_img = imread(img_path)[:, :, :3]  # drop alpha channel if it is provided
        print(f'[Reading Input Image] {time.time() - started_at:.2f} seconds')

        started_at = time.time()
        o2o_mask, _, m_inner = compute_masks(input_img)
        print(f'[Computing Masks] {time.time() - started_at:.2f} seconds')

        # make the input image prepared
        started_at = time.time()
        prep_input_img = srgb_to_prop_cat02(decode_srgb(to_single(input_img)))

        # choose a hint
        hints = {'none': None, 'o2o_all': o2o_mask, 'o2o_rgb': m_inner}  # all the hints here
        hint = hints[self.hint_mode]  # choose a particular hint using hint_mode

        if hint is not None:
            prep_hint_img = hint.astype(np.float32)  # type-matching
            network_input = np.dstack((prep_input_img, prep_hint_img))
        else:
            network_input = prep_input_img
            
        print(f'[Preparing Input Image] {time.time() - started_at:.2f} seconds')
        
        started_at = time.time()
        network_output = self.infer(network_input)
        print(f'[Inferring Output Image] {time.time() - started_at:.2f} seconds')

        started_at = time.time()
        network_output[o2o_mask] = prep_input_img[o2o_mask]  # use 'prep_input_img' at 'one-to-one' pixels
        output = encode_prop(network_output)  # encode ProPhoto RGB and clip from 0 to 1
        print(f'[Preparing Output Image] {time.time() - started_at:.2f} seconds')

        if output_img_path:
            started_at = time.time()
            imsave(output_img_path, to_uint8(output), optimize=False, compression=0, icc_profile=icc_profile)
            print(f'[Saving Output Image] {time.time() - started_at:.2f} seconds')

        return output

    def infer(self, network_input):
        network_input = to_tensor(network_input).to(self.device)  # to Tensor
        network_input = torch.unsqueeze(network_input, 0)  # add batch dimension
        with torch.no_grad():
            network_output = self.model(network_input.float()).detach()
        network_output = network_output.squeeze()  # remove batch dimension
        return network_output.transpose(0, 1).transpose(1, 2).cpu().numpy()  # to NumPy
    
    def gamut_expansion(self, srgb_img):
        o2o_mask, _, m_inner = compute_masks(srgb_img)
        # make the input image prepared
        prep_input_img = srgb_to_prop_cat02(srgb_img)

        # choose a hint
        hints = {'none': None, 'o2o_all': o2o_mask, 'o2o_rgb': m_inner}  # all the hints here
        hint = hints[self.hint_mode]  # choose a particular hint using hint_mode

        if hint is not None:
            prep_hint_img = hint.astype(np.float32)  # type-matching
            network_input = np.dstack((prep_input_img, prep_hint_img))
        else:
            network_input = prep_input_img
        
        started_at = time.time()
        network_output = self.infer(network_input)

        started_at = time.time()
        network_output[o2o_mask] = prep_input_img[o2o_mask]  # use 'prep_input_img' at 'one-to-one' pixels

        return network_output
        
    @property
    def hint_mode(self):
        return self.model.hparams.hint_mode
