import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.DOM import DOM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvTBNRelu(nn.Module):

    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



class UpConvt(nn.Module):


    def __init__(self, in_channels, out_channels, blocks):
        super(UpConvt, self).__init__()

        layers = [ConvTBNRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvTBNRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class DOM_fusion(nn.Module):
    def __init__(self, opt):
        super(DOM_fusion, self).__init__()
        #
        self.opt = opt
        self.batch_size = opt['train']['batch_size']
        self.msg_length = opt['network']['message_length']
        self.h, self.w = opt['network']['H'], opt['network']['W']
        down_opt = opt['network']['InvBlock']['downscaling']
        scale = pow(down_opt['scale'], down_opt['down_num'])
        self.scaled_h, self.scaled_w = int(opt['network']['H'] * scale), int(opt['network']['W'] * scale)
        self.diffusion_length = opt['network']['fusion']['fusion_length']  # 256
        blocks = opt['network']['fusion']['blocks']
        upconvT_channels = opt['network']['fusion']['upconvT_channels']
        # 
        self.linear1 = nn.Linear(opt['network']['message_length'], self.diffusion_length)
        self.linear2 = nn.Linear(opt['network']['message_length'], self.diffusion_length)
        self.linear3 = nn.Linear(opt['network']['message_length'], self.diffusion_length)
        self.reshape_length = int(math.sqrt(self.diffusion_length))  # 16
        #
        self.UpConvt1 = UpConvt(upconvT_channels, upconvT_channels, blocks)
        self.UpConvt2 = UpConvt(upconvT_channels, upconvT_channels, blocks)
        self.UpConvt3 = UpConvt(upconvT_channels, upconvT_channels, blocks)
        #
        self.linear_rev1 = nn.Linear(self.h * self.w, opt['network']['message_length'])
        self.linear_rev2 = nn.Linear(self.h * self.w, opt['network']['message_length'])
        self.linear_rev3 = nn.Linear(self.h * self.w, opt['network']['message_length'])
        #
        self.split1_img = opt['network']['InvBlock']['split1_img']
        self.repeat_num = opt['network']['InvBlock']['split2_repeat']
        self.decoder = DOM(self.h, self.w, self.msg_length).to(device)
        self.linear_expand = nn.Linear(30, 3 * 128 * 128)
        self.linear_compress = nn.Linear(3 * 128 * 128, 30)
        self.final_fc = nn.Linear(3 * 128 * 128, 30)


    def forward(self, image, message, haar, rev=False):
       
        if not rev:
            #
            expanded_message1 = self.linear1(message)
            expanded_message2 = self.linear2(message)
            expanded_message3 = self.linear3(message)
            #
            expanded_message1 = torch.reshape(expanded_message1,
                                              (expanded_message1.size(0), 1, self.reshape_length, self.reshape_length))
            expanded_message2 = torch.reshape(expanded_message2,
                                              (expanded_message2.size(0), 1, self.reshape_length, self.reshape_length))
            expanded_message3 = torch.reshape(expanded_message3,
                                              (expanded_message3.size(0), 1, self.reshape_length, self.reshape_length))
            #
            expanded_message1 = self.UpConvt1(expanded_message1)  # [b 1 16 16] ->  [b 1 128 128]
            expanded_message2 = self.UpConvt2(expanded_message2)
            expanded_message3 = self.UpConvt3(expanded_message3)
            #
            msg_concat = torch.cat([expanded_message1, expanded_message2, expanded_message3], dim=1)

            msg_down = haar(msg_concat)

            concat = torch.cat([image, msg_down], dim=1)

            return concat, msg_concat
        
        else:
            imgs = image[:, 0:self.split1_img, :, :]

            msg = image[:, self.split1_img:, :, :]
            msg = haar( msg , rev=True)

            msg_nms = self.decoder(msg)

            return imgs, msg_nms, msg
