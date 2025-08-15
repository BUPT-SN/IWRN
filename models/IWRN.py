import torch
import torch.nn as nn
from models.DAINN import DAIM
from models.Noise_pool import Noise_pool
from models.DOM_fusion import DOM_fusion
from models.LWN import InLWN
from models.NSM_module import NSM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FSM(nn.Module):
    def __init__(self, opt):
        super(FSM, self).__init__()
        self.split1_img = opt['network']['InvBlock']['split1_img']
        self.strength_factor = opt['noise']['StrengthFactor']['S']

    def forward(self, encoded_img, cover_down=None, rev=False):
        if not rev:
            msg = encoded_img[:, self.split1_img:, :, :]
            out = cover_down + self.strength_factor * msg
            return out
        else:
            out = torch.cat((encoded_img, encoded_img), dim=1)
            return out

class IWRNBasic(nn.Module):
    def __init__(self, opt, device):
        super(IWRNBasic, self).__init__()
    
        self.opt = opt
        self.device = device
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        self.invertible_model = DAIM(opt).to(device)
        self.cs_model = FSM(opt).to(device)
        self.noise_model = Noise_pool(opt, device).to(device)
        self.fusion_model = DOM_fusion(opt).to(device)
        if opt['network']['InvBlock']['downscaling']['use_down']:
            self.invDown = InLWN(opt).to(device)



        self.nsm_model = NSM(opt, self.device)
    def encoder(self, image, msg):

        cover_down = self.invDown(image) 
        fusion, residual = self.fusion_model(cover_down, msg, self.invDown) 
        inv_encoded = self.invertible_model(fusion)
        cs = self.cs_model(inv_encoded, cover_down)
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1) 

        return watermarking_img, residual

    def noise_pool(self, watermarking_img, image, noise_choice):
        noised_img = self.noise_model(watermarking_img, image, noise_choice) 
        return noised_img

    def super_noise_pool(self, watermarking_img, image, noise_choice='Superposition'):
        super_noised_img = self.noise_model(watermarking_img, image, noise_choice) 
        return super_noised_img
    def nsm(self, noised_img):
        return torch.round(
            torch.mean((torch.argmax(self.nsm_model(noised_img.clone().detach().clamp(-1, 1)), dim=1)).float()))

   
    def train_val_decoder(self, watermarking_img, super_noised_img, noised_img, noise_choice):
        down = self.invDown(noised_img)  
        cs_rev = self.cs_model(down, rev=True)
        inv_back = self.invertible_model(cs_rev, rev=True)  
        img_fake, msg_fake, residual = self.fusion_model(inv_back, None, self.invDown, rev=True)  
        img_fake = self.invDown(img_fake, rev=True)  
        img_fake = img_fake.clamp(-1, 1)
        msg_nsm = None
        return img_fake, msg_fake, msg_nsm, residual


    def test_decoder(self, noised_img, pre_noise):
        down = self.invDown(noised_img)  # [64]
        cs_rev = self.cs_model(down, rev=True)
        inv_back = self.invertible_model(cs_rev, rev=True)  # [64]
        img_fake, msg_fake,_ = self.fusion_model(inv_back, None, self.invDown, rev=True)  # [64]
        img_fake = self.invDown(img_fake, rev=True)  # [128]
        img_fake = img_fake.clamp(-1, 1)
        msg_nsm = msg_fake
        return img_fake, msg_fake, msg_nsm


class IWRN(IWRNBasic):
    def __init__(self, opt, device):
        super(IWRN, self).__init__(opt, device)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, image, msg, noise_choice, is_train=True):
        if is_train:
            watermarking_img ,residual1= self.encoder(image, msg)
            noised_img = self.noise_pool(watermarking_img.clone(), image, noise_choice)
            img_fake, msg_fake, msg_nsm, residual2 = self.train_val_decoder(
                watermarking_img, image, noised_img, noise_choice)
            out_feature = self.avgpool(image)
        else:
            watermarking_img ,residual1= self.encoder(image, msg)
            noised_img = self.noise_pool(watermarking_img.clone(), image, noise_choice)
            pre_noise = self.nsm(noised_img)
            img_fake, msg_fake, msg_nsm = self.test_decoder(noised_img, pre_noise)
            out_feature = self.avgpool(image)
            residual1 = None
            residual2 = None

        return image, out_feature, watermarking_img, noised_img, img_fake, msg_fake, msg_nsm,residual1,residual2


