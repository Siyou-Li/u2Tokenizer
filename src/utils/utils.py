import re
import torch
import numpy as np

def mask2box(mask_3d):
    non_zero_indices = torch.nonzero(mask_3d, as_tuple=True)

    min_indices = [torch.min(indices) for indices in non_zero_indices]
    max_indices = [torch.max(indices) for indices in non_zero_indices]

    min_x = min_indices[0].item()
    min_y = min_indices[1].item()
    min_z = min_indices[2].item()
    max_x = max_indices[0].item()
    max_y = max_indices[1].item()
    max_z = max_indices[2].item()

    X, Y, Z = mask_3d.shape

    min_x_norm = round(min_x / X, 3)
    min_y_norm = round(min_y / Y, 3)
    min_z_norm = round(min_z / Z, 3)

    max_x_norm = round(max_x / X, 3)
    max_y_norm = round(max_y / Y, 3)
    max_z_norm = round(max_z / Z, 3)

    return [min_x_norm, min_y_norm, min_z_norm, max_x_norm, max_y_norm, max_z_norm]


def extract_box_from_text(text):
    match = re.search(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    if match:
        box_coordinates = [float(coord) for coord in match.groups()]
        return box_coordinates
    else:
        return None

def calculate_iou(box1, box2):
    x1, y1, z1, x2, y2, z2 = box1
    x3, y3, z3, x4, y4, z4 = box2

    inter_x = max(0, min(x2, x4) - max(x1, x3))
    inter_y = max(0, min(y2, y4) - max(y1, y3))
    inter_z = max(0, min(z2, z4) - max(z1, z3))

    union_x = max(x2, x4) - min(x1, x3)
    union_y = max(y2, y4) - min(y1, y3)
    union_z = max(z2, z4) - min(z1, z3)

    vol_intersection = inter_x * inter_y * inter_z
    vol_union = union_x * union_y * union_z

    iou = vol_intersection / (vol_union + 1e-6)
    return iou

def normalize(ct_tensor):
    ct_numpy = ct_tensor.numpy()
    ct_voxel_numpy = ct_numpy.copy().flatten()
    
    thred = np.mean(ct_voxel_numpy)
    voxel_filtered = ct_voxel_numpy[ct_voxel_numpy > thred]
    
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 00.05)

    mean = np.mean(voxel_filtered)
    std  = np.std(voxel_filtered)
    
    ct_numpy = np.clip(ct_numpy, lower_bound, upper_bound)
    ct_numpy = (ct_numpy - mean) / max(std, 1e-8)
    
    ct_numpy = ct_numpy - np.min(ct_numpy)
    ct_numpy = ct_numpy / max(np.max(ct_numpy), 1e-8)
    
    ct_numpy = np.expand_dims(ct_numpy, axis=0)
    ct_numpy = np.transpose(ct_numpy, (0, 3, 1, 2))
    
    return ct_numpy

# def normalize(ct_tensor):
#     # Flatten a copy of the tensor to work with voxel values
#     ct_voxel = ct_tensor.view(-1)
    
#     # Compute the threshold as the mean of all voxel values
#     threshold = torch.mean(ct_voxel)
    
#     # Filter out only the voxel values that are above the threshold
#     voxel_filtered = ct_voxel[ct_voxel > threshold]
    
#     # Compute the lower and upper bounds based on percentiles.
#     # Note: np.percentile with 0.05 corresponds to the 0.0005 quantile (i.e. 0.05/100)
#     # and 99.95 corresponds to the 0.9995 quantile.
#     lower_bound = torch.quantile(voxel_filtered, 0.0005)
#     upper_bound = torch.quantile(voxel_filtered, 0.9995)
    
#     # Compute the mean and std of the filtered voxels
#     filtered_mean = torch.mean(voxel_filtered)
#     filtered_std = torch.std(voxel_filtered)
    
#     # Clip the values of the tensor to be within the computed bounds
#     ct_tensor = torch.clamp(ct_tensor, min=lower_bound, max=upper_bound)
    
#     # Normalize using the filtered mean and std (guarding against division by zero)
#     ct_tensor = (ct_tensor - filtered_mean) / torch.clamp(filtered_std, min=1e-8)
    
#     # Shift the tensor so its minimum value is 0
#     ct_tensor = ct_tensor - torch.min(ct_tensor)
    
#     # Scale the tensor so that its maximum value is 1 (again, avoiding division by zero)
#     ct_tensor = ct_tensor / torch.clamp(torch.max(ct_tensor), min=1e-8)
    
#     # Optionally, if you need to add a channel dimension and transpose,
#     # you can uncomment the following lines:
#     # ct_tensor = ct_tensor.unsqueeze(0)
#     # ct_tensor = ct_tensor.permute(0, 3, 1, 2)
    
#     return ct_tensor