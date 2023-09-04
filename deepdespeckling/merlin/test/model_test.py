import pathlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from deepdespeckling.merlin.test.model import *
from deepdespeckling.merlin.test.utils import *

M = 10.089038980848645
m = -1.429329123112601


class Denoiser(object):
    """ Description
                ----------
                A set of initial conditions, and transformations on the Y

                Parameters
                ----------
                denoiser : an object

                Returns
                ----------
    """

    def __init__(self, input_c_dim=1):

        self.input_c_dim = input_c_dim

    def load(self, model, weights_path):
        """ Description
                    ----------
                    Restores a checkpoint located in a checkpoint repository

                    Parameters
                    ----------
                    checkpoint_dir : a path leading to the checkpoint file

                    Returns
                    ----------
                    True : Restoration is a success
                    False: Restoration has failed
        """
        print("[*] Loading the model...")

        model.load_state_dict(torch.load(weights_path))

        return model

    def test(self, images, weights_path, stride, patch_size, device="cuda:0"):
        """ Description
                    ----------
                    The function that does the job. Should be merged with main.py ?

                    Parameters
                    ----------
                    test_files : a path leading to the checkpoint file
                    ckpt_dir : repository containing the checkpoint (and weights)
                    dataset_dir : the path to the test data
                    stride : number of bytes from one row of pixels in memory to the next row of pixels in memory

                    Returns
                    ----------
                    True : Restoration is a success
                    False: Restoration has failed

        """

        """Test MERLIN"""

        device = torch.device(device if torch.cuda.is_available() else "cpu")

        loaded_model = Model(device)
        loaded_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        loaded_model.to(device)

        # loaded_model = self.load(model,weights_path)
        outputs = []

        for real_image in images:
            real_image = np.array(real_image).reshape(1, np.size(real_image, 0), np.size(real_image, 1), 2).astype(np.float32)
            i_real_part = (real_image[:, :, :, 0]).reshape(real_image.shape[0], real_image.shape[1],
                                                           real_image.shape[2], 1)
            i_imag_part = (real_image[:, :, :, 1]).reshape(real_image.shape[0], real_image.shape[1],
                                                           real_image.shape[2], 1)

            # Pad the image
            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(i_real_part.shape)
            output_clean_image_1 = np.zeros(i_real_part.shape)
            output_clean_image_2 = np.zeros(i_real_part.shape)

            if im_h == patch_size:
                x_range = list(np.array([0]))

            else:
                x_range = list(range(0, im_h - patch_size, stride))
                if (x_range[-1] + patch_size) < im_h:
                    x_range.extend(range(im_h - patch_size, im_h - patch_size + 1))

            if im_w == patch_size:
                y_range = list(np.array([0]))

            else:
                y_range = list(range(0, im_w - patch_size, stride))
                if (y_range[-1] + patch_size) < im_w:
                    y_range.extend(range(im_w - patch_size, im_w - patch_size + 1))

            for x in tqdm(x_range):
                for y in y_range:

                    real_to_denoise, imag_to_denoise = symetrisation_patch_test(
                        i_real_part[:, x:x + patch_size, y:y + patch_size, :], i_imag_part[:, x:x + patch_size, y:y + patch_size, :])

                    real_to_denoise = torch.tensor(real_to_denoise, device=device)
                    imag_to_denoise = torch.tensor(imag_to_denoise, device=device)

                    real_to_denoise = real_to_denoise.type(torch.float32)
                    imag_to_denoise = imag_to_denoise.type(torch.float32)

                    real_to_denoise = (torch.log(torch.square(real_to_denoise)+1e-3)-2*m)/(2*(M-m))
                    imag_to_denoise = (torch.log(torch.square(imag_to_denoise)+1e-3)-2*m)/(2*(M-m))

                    tmp_clean_image_real = loaded_model.forward(real_to_denoise).detach().to('cpu').numpy()
                    tmp_clean_image_real = np.moveaxis(tmp_clean_image_real, 1, -1)

                    output_clean_image_1[:, x:x + patch_size, y:y + patch_size, :] = output_clean_image_1[:, x:x + patch_size,
                                                                                                          y:y + patch_size,
                                                                                                          :] + tmp_clean_image_real

                    tmp_clean_image_imag = loaded_model.forward(imag_to_denoise).detach().to('cpu').numpy()
                    tmp_clean_image_imag = np.moveaxis(tmp_clean_image_imag, 1, -1)

                    output_clean_image_2[:, x:x + patch_size, y:y + patch_size, :] = output_clean_image_2[:, x:x + patch_size,
                                                                                                          y:y + patch_size,
                                                                                                          :] + tmp_clean_image_imag
                    count_image[:, x:x + patch_size, y:y + patch_size, :] = count_image[:, x:x + patch_size, y:y + patch_size,
                                                                                        :] + np.ones((1, patch_size, patch_size, 1))

            output_clean_image_1 = output_clean_image_1 / count_image
            output_clean_image_2 = output_clean_image_2 / count_image
            output_clean_image = 0.5 * (np.square(denormalize_sar(output_clean_image_1)) + np.square(
                denormalize_sar(output_clean_image_2)))  # combine the two estimation

            outputimage = np.sqrt(np.squeeze(output_clean_image))

            noisyimage = np.squeeze(np.sqrt(i_real_part ** 2 + i_imag_part ** 2))

            threshold = np.mean(noisyimage) + 5 * np.std(noisyimage)
            denoised_clip = np.clip(outputimage, 0, threshold)
            denoised_clip = denoised_clip / threshold * 255
            
            outputs.append(outputimage)

        return outputs
