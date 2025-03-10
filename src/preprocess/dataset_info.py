import os
from monai.transforms import (
        LoadImage,
        Compose,
)
from monai.data.image_reader import NibabelReader

transforms = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
    ]
)
image_dir = "/import/c4dm-04/siyoul/u2Tokenizer/datasets/AMOS-MM/imagesVa"
nii_files = os.listdir(image_dir)

# Statistics of shape distribution
shapes = []
for file in nii_files:
    input_path = os.path.join(image_dir, file)
    data = transforms(input_path)
    shape = data.shape
    shapes.append(shape)

print(f"[*] Number of data: {len(shapes)}")
# max size of each dimension
max_shape = [max([shape[i] for shape in shapes]) for i in range(3)]
print(f"[*] Max shape: {max_shape}")
# min size of each dimension
min_shape = [min([shape[i] for shape in shapes]) for i in range(3)]
print(f"[*] Min shape: {min_shape}")
