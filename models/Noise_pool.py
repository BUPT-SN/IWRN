import torch.nn as nn
from models.All_Noise import Identity, ColorJitter, Cropout, Resize, GaussianNoise, Salt_Pepper,GaussianBlur, Dropout
import kornia
import random


class Noise_pool(nn.Module):


    def __init__(self, opt, device):
        super(Noise_pool, self).__init__()
        #
        self.opt = opt
        self.si_pool = opt["noise"]["Superposition"]["si_pool"]
        self.one_input_parms = ["Rotation", "Affine"]
        #
        self.Identity = Identity()
        self.Resize = Resize(opt)
        self.GaussianBlur = GaussianBlur(opt)
        self.Salt_Pepper = Salt_Pepper(opt, device)
        self.GaussianNoise = GaussianNoise(opt, device)
        self.Brightness = ColorJitter(opt, distortion="Brightness")
        self.Contrast = ColorJitter(opt, distortion="Contrast")
        self.Saturation = ColorJitter(opt, distortion="Saturation")
        self.Hue = ColorJitter(opt, distortion="Hue")
        self.Rotation = kornia.augmentation.RandomRotation(
            degrees=opt["noise"]["Rotation"]["degrees"],
            p=opt["noise"]["Rotation"]["p"],
            keepdim=True,
        )
        self.Affine = kornia.augmentation.RandomAffine(
            degrees=opt["noise"]["Affine"]["degrees"],
            translate=opt["noise"]["Affine"]["translate"],
            scale=opt["noise"]["Affine"]["scale"],
            shear=opt["noise"]["Affine"]["shear"],
            p=opt["noise"]["Affine"]["p"],
            keepdim=True,
        )
        #
        self.Cropout = Cropout(opt)
        self.Dropout = Dropout(opt)

    def forward(self, encoded, cover_img, noise_choice):
        if noise_choice == "Superposition":
            noised_img = self.Superposition(encoded, cover_img)
        else:
            noised_img = (
                self.noise_pool()[noise_choice](encoded)
                if noise_choice in self.one_input_parms
                else self.noise_pool()[noise_choice](encoded, cover_img)
            )
        return noised_img

    def Superposition(self, encoded, cover_img):
        si_pool = self.si_pool
        random.shuffle(self.si_pool) if self.opt["noise"]["Superposition"][
            "shuffle"
        ] else None
        for key in si_pool:
            encoded = (
                self.noise_pool()[key](encoded)
                if key in ["Rotation", "Affine"]
                else self.noise_pool()[key](encoded, cover_img)
            )
        return encoded

    def Space(self, wm_imgs, cover_img=None):
        rand_num = random.randint(0, 1)

        if rand_num == 0:
            transformed_imgs = self.Rotation(wm_imgs)
        else:
            transformed_imgs = self.Affine(wm_imgs)

        return transformed_imgs

    def noise_pool(self):
        return {
            "Identity": self.Identity,
            "Resize": self.Resize,
            "GaussianBlur": self.GaussianBlur,
            "Salt_Pepper": self.Salt_Pepper,
            "GaussianNoise": self.GaussianNoise,
            "Brightness": self.Brightness,
            "Contrast": self.Contrast,
            "Saturation": self.Saturation,
            "Hue": self.Hue,
            "Rotation": self.Rotation,
            "Affine": self.Affine,
            "Cropout": self.Cropout,
            "Dropout": self.Dropout,
            "Space": self.Space,

        }
