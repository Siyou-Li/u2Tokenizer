import nibabel as nib
import torch.nn.functional as F
from monai.data.utils import correct_nifti_header_if_necessary
import torch
import numpy as np

class NIfTIProcessor:
    def __init__(self, output_size, patch_size, mode):
        """
        Initialize the NiiProcessor class.
        
        Args:
        - output_size (tuple): Target size of the output image (D, H, W).
        - patch_size (tuple): Size of the patches to be extracted (d, h, w).
        - mode (str): Interpolation mode, e.g., 'trilinear'.
        """
        self.output_size = output_size
        self.patch_size = patch_size
        self.mode = mode

    def load_nii_file(self, file_path):
        """
        Load the NIfTI file and return the image data.
        
        Args:
        - file_path (str): Path to the NIfTI file.
        
        Returns:
        - img_data (numpy.ndarray): Loaded image data.
        """
        img = nib.load(file_path)
        img = correct_nifti_header_if_necessary(img)
        img_data = img.get_fdata()
        return img_data

    def scale_image(self, file_path):
        """
        Scale the image to the target size.
        
        Returns:
        - warp_img (torch.Tensor): Scaled image tensor.
        """
        img_data = self.load_nii_file(file_path)
        # Normalize the image data
        # mean = np.mean(img_data)
        # std = np.std(img_data)
        # img_data = (img_data - mean) / std
        max_val = np.max(img_data)
        min_val = np.min(img_data)
        img_data = (img_data - min_val) / (max_val - min_val)

        img_data = torch.from_numpy(img_data)
        img_data = img_data.permute(2, 0, 1)  # shape: (D, H, W)
        new_D, new_H, new_W = self.output_size
        if self.output_size[0] == -1:
            new_D = img_data.shape[0]
        warp_img = img_data.unsqueeze(0).unsqueeze(0) 
        warp_img = F.interpolate(warp_img, size=(new_D, new_H, new_W), mode=self.mode, align_corners=True)
        warp_img = warp_img.squeeze(0).squeeze(0)
        return warp_img