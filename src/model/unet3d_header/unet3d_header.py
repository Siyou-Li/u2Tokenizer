import torch
import torch.nn.functional as F

from .UNet import UNet3D

class ResBlock3D(torch.nn.Module):
    def __init__(self, in_channels=1, f_maps=16, out_channels=1):
        super(ResBlock3D, self).__init__()
        self.f_maps = f_maps
        
        self.conv1 = torch.nn.Conv3d(in_channels, self.f_maps, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv3d(self.f_maps, self.f_maps, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(self.f_maps, out_channels, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(F.relu(y))
        z = x + y
        return self.conv3(F.relu(z))

# class Unet3DHead(torch.nn.Module):
#     def __init__(self, max_resolution=256, output_shape = (32, 256, 256)):
#         super(Unet3DHead, self).__init__()
#         self.in_channels = 1
#         self.out_channels = 1
#         self.max_resolution = max_resolution
#         self.f_maps = 32
#         self.num_levels = 4
#         self.num_groups = 8
#         self.output_shape = output_shape
#         self.dtype = torch.float32

#         self.UNet3D = ResBlock3D(self.f_maps, self.num_groups)

#         # self.UNet3D = UNet3D(in_channels=self.in_channels, out_channels=self.out_channels, f_maps=self.f_maps, reshape = self.output_shape)
#         self.UNet3D = self.UNet3D.to(dtype=self.dtype)

#     def forward(self, x):
#         assert len(x.shape) == 5
#         x_shape = torch.tensor(x.shape[-3:])
#         x = x.to(dtype=self.dtype)
#         ## clip the input image to the max resolution
#         x_shape = torch.where(x_shape > self.max_resolution, self.max_resolution, x_shape)
#         x_shape = (x_shape[0].item(), x_shape[1].item(), x_shape[2].item())
        
#         # x = F.interpolate(x, x_shape, mode='trilinear')
#         x 
#         x = self.UNet3D(x)['final']
#         x = self.UNet3D(x)
        
#         return x

# class Unet3DHead(torch.nn.Module):
#     def __init__(self, max_resolution=128, output_shape = (32, 256, 256)):
#         super(Unet3DHead, self).__init__()
#         self.in_channels = 1
#         self.out_channels = 1
#         self.max_resolution = max_resolution
#         self.f_maps = 32
#         self.num_levels = 4
#         self.output_shape = output_shape
#         self.dtype = torch.float32
#         # self.UNet3D = UNet3D(in_channels=self.in_channels, out_channels=self.out_channels, f_maps=self.f_maps, reshape = self.output_shape)
#         # self.UNet3D = self.UNet3D.to(dtype=self.dtype)
#         self.head = torch.nn.Sequential(
#             torch.nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0),
#             torch.nn.Sigmoid()
#         )
#         self.UNet3D = ResBlock3D(f_maps=self.f_maps)

#     def forward(self, x):
#         assert len(x.shape) == 5
#         x_shape = torch.tensor(x.shape[-3:])
#         x = x.to(dtype=self.dtype)
#         ## clip the input image to the max resolution
#         x_shape = torch.where(x_shape > self.max_resolution, self.max_resolution, x_shape)
#         x_shape = (x_shape[0].item(), x_shape[1].item(), x_shape[2].item())
        
#         x = self.head(x)
#         x = F.adaptive_avg_pool3d(x, x_shape)
#         x = self.UNet3D(x)
#         x = F.adaptive_max_pool3d(x, self.output_shape)
#         return x

class Conv3DHead(torch.nn.Module):
    def __init__(self, max_resolution=128, output_shape = (32, 256, 256)):
        super(Conv3DHead, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.output_shape = output_shape
        self.dtype = torch.float32
        self.head = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        assert len(x.shape) == 5
        x = self.head(x)
        x = F.adaptive_max_pool3d(x, self.output_shape)
        return x

if __name__ == "__main__":
    conv3d_head = Conv3DHead()
    input_tensor = torch.randn(32, 1, 64, 256, 256)
    output_tensor = conv3d_head(input_tensor)
    print(output_tensor.shape)