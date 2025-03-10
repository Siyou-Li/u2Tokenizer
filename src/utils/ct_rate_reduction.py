# -*- encoding: utf-8 -*-
# @File        :   ct_rate_resize.py
# @Time        :   2025/01/13 17:41:14
# @Author      :   Siyou
# @Description :

from monai.data.image_reader import NibabelReader
from monai.transforms import (
        LoadImage,
        Compose,
        CropForeground,
        SaveImage,
    )
from monai.transforms.spatial.functional import resize
import os
import shutil
#from config import config

base_path = "/import/c4dm-04/siyoul/u2Tokenizer"
class NlfTIUtils:
    def __init__(self, mode='trilinear'):
        self.adaptive_transforms = Compose(
            [
                LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                CropForeground(source_key="image"),
            ]
        )
        self.mode = mode
        
        self.save = SaveImage(separate_folder=False, output_postfix='')
    def NlfTI_adaptive_resize(self, input_path, output_path, target_size=[256, 256, 256]):
        """
        adaptive resize the NIfTI file to the target size.
        The minimum dimension is scaled to the target dimension, and other dimensions are scaled proportionally
        """
        data = self.adaptive_transforms(input_path)
        input_shape = data.shape
        ratio = max([target_size[i] / input_shape[i] for i in range(2)])
        if ratio >= 1:
            print(f"Skip {input_path}")
            return
        target_shape = [int(input_shape[i] * ratio) for i in range(3)]
        data = resize(
            img=data.unsqueeze(0), 
            out_size=target_shape, 
            mode=self.mode,
            align_corners=True,
            dtype=None,
            input_ndim=3,
            anti_aliasing= True,
            anti_aliasing_sigma=None,
            lazy=False,
            transform_info=None,
            )
        self.save(data[0], output_path)
        shutil.move(
            os.path.join(base_path,"datasets", os.path.basename(output_path)), output_path
            )

def array_split(raw_list:list, split_num:int)->list:
    print("split length:",len(raw_list))
    avg = len(raw_list) / float(split_num)
    split_list = []
    last = 0.0
    while last < len(raw_list):
        split_list.append(raw_list[int(last):int(last + avg)])
        last += avg
    return split_list

if __name__ == "__main__":
    
    import os
    from multiprocessing import Process
    
    image_dir = os.path.join(base_path, "datasets/CT-RATE/dataset/train")

    num_workers = 16
    def worker(work_id, sub_image_dir, image_dir=image_dir):
        nifti_utils = NlfTIUtils()
        
        for dir in sub_image_dir:
            nii_dirs_1 = os.listdir(os.path.join(image_dir, dir))
            for dir_1 in nii_dirs_1:
                nii_dirs_2 = os.listdir(os.path.join(image_dir, dir, dir_1))
                for file in nii_dirs_2:
                    path = os.path.join(image_dir, dir, dir_1, file)
                    print(f"worker-{work_id} Processing {path}")
                    try:
                        nifti_utils.NlfTI_adaptive_resize(path, path)
                    except Exception as e:
                        print(e)
                        print(path)
                        continue
    # split the image dir
    sub_image_dirs = array_split(os.listdir(image_dir), num_workers)
    print("Number of image dirs:", len(os.listdir(image_dir)))
    process_list = []
    for i, sub_image_dir in enumerate(sub_image_dirs):
        p = Process(target=worker, args=(i, sub_image_dir))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()