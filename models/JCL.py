from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
import torch


class JCLD(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        num_ftrs = 3
        proj_hidden_dim = 512
        out_dim = 16  
        pred_hidden_dim = 256 
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x, message,noise_choice,rev):
        out_put, out_feature ,watermarking_img, noised_img, img_fake, msg_fake, _, residual1,residual2= self.backbone(x, message,noise_choice,rev)
        f = out_feature.flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()

        return z, p, watermarking_img, noised_img, img_fake, msg_fake, _, residual1,residual2
    

class JCLE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_ftrs = 3
        proj_hidden_dim = 512
        out_dim = 16 
        pred_hidden_dim = 256 
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = SimSiamProjectionHead(
                num_ftrs, proj_hidden_dim, out_dim
            )
        self.prediction_head = SimSiamPredictionHead(
                out_dim, pred_hidden_dim, out_dim
            )

    def forward(self, image):
        out_feature = self.avgpool(image)
        f = out_feature.flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)

        z = z.detach()

        return z, p