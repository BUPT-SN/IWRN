import torch.nn as nn
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
import torchvision.transforms as transforms
from typing import Tuple
#
class Salt_Pepper(nn.Module):
 

    def __init__(self, opt, device):
        super(Salt_Pepper, self).__init__()
        # pepper and salt
        self.snr = opt['noise']['Salt_Pepper']['snr']
        self.p = opt['noise']['Salt_Pepper']['p']
        self.device = device

    def forward(self, encoded, cover_img):
        if random.uniform(0, 1) < self.p: 
            #
            b, c, h, w = encoded.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = torch.Tensor(np.random.choice((0, 1, 2), size=(b, 1, h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])).to(self.device)
            mask = mask.repeat(1, c, 1, 1)
            #
            encoded[mask == 1] = 1      # salt
            encoded[mask == 2] = -1     # pepper

            return encoded
        else:
            print('salt_pepper error!')
            exit()

class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, opt):
        super(Resize, self).__init__()
        resize_ratio_down = opt['noise']['Resize']['p']
        self.h = opt['network']['H']
        self.w = opt['network']['W']
        self.scaled_h = int(resize_ratio_down * self.h)
        self.scaled_w = int(resize_ratio_down * self.w)
        self.interpolation_method = "bicubic"
        # self.interpolation_method = "nearest"
    def forward(self, wm_imgs, cover_img=None):
        
        #
        noised_down = F.interpolate(
                                    wm_imgs,
                                    size=(self.scaled_h, self.scaled_w),
                                    mode=self.interpolation_method
                                    )
        noised_up = F.interpolate(
                                    noised_down,
                                    size=(self.h, self.w),
                                    mode=self.interpolation_method
                                    )

        return noised_up

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, encoded, cover_img=None):
        out = encoded.clone()
        return out

class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        # gaussian
        self.mean = opt['noise']['GaussianNoise']['mean']
        self.variance =  opt['noise']['GaussianNoise']['variance']
        self.amplitude =  opt['noise']['GaussianNoise']['amplitude']
        self.p = opt['noise']['GaussianNoise']['p']
        self.device = device

    def forward(self, encoded, cover_img=None):
        if random.uniform(0, 1) < self.p:
            #
            b, c, h, w = encoded.shape
            #
            Noise = self.amplitude * torch.Tensor(np.random.normal(loc=self.mean, scale=self.variance, size=(b, 1, h, w))).to(self.device)
            Noise = Noise.repeat(1, c, 1, 1)
            #
            img_ = Noise + encoded

            return img_

        else:
            print('Gaussian noise error!')
            exit()

def gaussian(window_size, sigma):

    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)

    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:

    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class GaussianBlur(nn.Module):

    def __init__(self, opt):
        super(GaussianBlur, self).__init__()
        
        kernel_size = (opt['noise']['GaussianBlur']['kernel_sizes'], opt['noise']['GaussianBlur']['kernel_sizes'])
        sigma = (opt['noise']['GaussianBlur']['sigmas'], opt['noise']['GaussianBlur']['sigmas'])
        
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor, cover_img):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)


        out = conv2d(x, kernel, padding=self._padding, stride=1, groups=c)
        return out
    
class Dropout(nn.Module):
 
    def __init__(self, opt):
        super(Dropout, self).__init__()
        #
        self.p = opt['noise']['Dropout']['p']

    def forward(self, encoded_img, cover_image):
        

        mask = np.random.choice([0.0, 1.0], encoded_img.shape[2:], p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=encoded_img.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(encoded_img)
        noised_image = encoded_img * mask_tensor + cover_image * (1-mask_tensor)
        
        return noised_image
    
def random_float(min, max):

    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):

    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(height_ratio_range * image_height)
    remaining_width = int(width_ratio_range * image_width)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Cropout(nn.Module):

    def __init__(self, opt):

        super(Cropout, self).__init__()
        #
        ratio = 1 - opt['noise']['Cropout']['p']  # 0.5 means 50% retain of total watermarked pixel
        #
        self.height_ratio_range = int(np.sqrt(ratio) * 100) / 100  
        self.width_ratio_range = int(np.sqrt(ratio) * 100) / 100


    def forward(self, encoded_img, cover_img):
        
        #
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(encoded_img, self.height_ratio_range, self.width_ratio_range)
        #
        out = cover_img.clone()
        out[:,:, h_start:h_end, w_start: w_end] = encoded_img[:,:, h_start:h_end, w_start: w_end]        
        
        return out
    

#
class ColorJitter(nn.Module):

    def __init__(self, opt, distortion):
        super(ColorJitter, self).__init__()
        #
        brightness   = opt['noise']['Brightness']['f']
        contrast     = opt['noise']['Contrast']['f']
        saturation   = opt['noise']['Saturation']['f']
        hue          = opt['noise']['Hue']['f']
        #
        self.distortion = distortion
        self.transform = None

        if distortion == 'Brightness':
            self.transform = transforms.ColorJitter(brightness=brightness)
        if distortion == 'Contrast':
            self.transform = transforms.ColorJitter(contrast=contrast)
        if distortion == 'Saturation':
            self.transform = transforms.ColorJitter(saturation=saturation)
        if distortion == 'Hue':
            self.transform = transforms.ColorJitter(hue=hue)

    def forward(self, watermarked_img, cover_img=None):
        #
        watermarked_img = (watermarked_img + 1 ) / 2   # [-1, 1] -> [0, 1]
        ColorJitter = self.transform(watermarked_img)
        ColorJitter = (ColorJitter * 2) - 1  # [0, 1] -> [-1, 1]

        return ColorJitter
