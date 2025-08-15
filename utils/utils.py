import os
import csv
import logging
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch.nn.functional as F
from PIL import ImageFile
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image


class WM_Dataset(Dataset):
    def __init__(self, opt):
        super(WM_Dataset, self).__init__()
        if opt['train/test'] == 'train':
            self.num_of_load = opt['datasets']['nDatasets']['num']
            path = opt['path']['train_folder']
        else:   
            self.num_of_load = opt['datasets']['test']['num']
            path = opt['path']['test_folder']
        #
        imgs=os.listdir(path)
        self.imgs=[os.path.join(path,k) for k in imgs]
        self.input_transforms = transforms.Compose([
            transforms.CenterCrop((opt['datasets']['H'], opt['datasets']['W'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
    def __getitem__(self, index):
        #
        data = self.imgs[index]
        img = Image.open(data)   
        img = img.convert('RGB') 
        img = self.input_transforms(img)  
        return img

    def __len__(self):
        return self.num_of_load


def train_val_loaders(opt):
    data_input = WM_Dataset(opt)
    train_size = int(opt['datasets']['nDatasets']['nTrain'] * len(data_input))
    test_size = len(data_input) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_input, [train_size, test_size])
    train_loader = DataLoader(dataset = train_dataset, batch_size=opt['train']['batch_size'], shuffle=True, num_workers=opt['train']['num_workers'])
    val_loader = DataLoader(dataset = test_dataset,  batch_size=opt['train']['batch_size'], shuffle=False, num_workers=opt['train']['num_workers'])
    
    return train_loader, val_loader


def test_loader(opt):

    data_input = WM_Dataset(opt)
    test_loader = DataLoader(dataset = data_input,  batch_size=opt['train']['batch_size'], shuffle=False, num_workers=opt['train']['num_workers'])
    return test_loader


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def write_losses(file_name, epoch, loss1=0, loss2=0, loss3=0):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = ['{:.0f}'.format(epoch)] + ['{:.4f}'.format(loss1)] + ['{:.4f}'.format(loss2)] + ['{:.4f}'.format(loss3)]
        writer.writerow(row_to_write)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_image(
    tensor,
    fp,
    nrow: int = 16,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format = None,
) -> None:

    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, scale_each=scale_each)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)



def _normalize(input_tensor):
    min_val, max_val = torch.min(input_tensor), torch.max(input_tensor)
    return (input_tensor-min_val) / (max_val-min_val)



def save_images(protected_img, watermarking_img, noised_img, img_fake, epoch, current_step, folder, time_now_NewExperiment, opt, resize_to=None):
    
    protected_img = protected_img.cpu()
    watermarking_img = watermarking_img.cpu()
    noised_img = noised_img.cpu()
    img_fake = img_fake.cpu()
    protected_img = (protected_img + 1) / 2
    watermarking_img = (watermarking_img + 1) / 2
    noised_img = (noised_img + 1) / 2
    diff_w2po = _normalize(torch.abs(protected_img - watermarking_img))
    diff_w2no = _normalize(torch.abs(noised_img - watermarking_img))
    
    if resize_to is not None:
        protected_img = F.interpolate(protected_img, size=resize_to)
        watermarking_img = F.interpolate(watermarking_img, size=resize_to)
        diff_w2po = F.interpolate(diff_w2po, size=resize_to)
        diff_w2no = F.interpolate(diff_w2no, size=resize_to)
    #
    stacked_images = torch.cat([protected_img, watermarking_img, noised_img, diff_w2po, diff_w2no], dim=0)
    filename = os.path.join(folder, 'epoch-{}-step-{}-{}.png'.format(epoch, current_step, time_now_NewExperiment))
    saveFormat = opt['train']['saveFormat']
    if opt['train']['saveStacked']:
        save_image(stacked_images, filename + saveFormat, protected_img.shape[0], normalize=False)
    else:
        save_image(watermarking_img, filename + '-watermarking' + saveFormat, normalize=False)




def save_tensor_images(input, folder_name=None):
    #
    img0 = input[0,:,:,:]
    img0 = img0.unsqueeze(0)
    img0 = img0.cpu()
    img0 = (img0 + 1) / 2
    #
    img0 = img0.reshape(img0.shape[1],img0.shape[0],img0.shape[2], img0.shape[3])
    #
    folder = '/.../debug/{}'.format(folder_name)
    mkdir(folder)
    saveFormat = '.png'
    stacked_images = img0
    save_image(stacked_images, folder + saveFormat, input.shape[1], normalize=False)


#
def func_loss_RecMsg(RecMsgLoss, message, msg_fake_1):
    
    loss_RecMsg  =  RecMsgLoss(message, msg_fake_1)


    return loss_RecMsg

def loss_lamd(current_step, opt):
    #
    if opt['loss']['option'] == 'lamd':
    
        lw_Rec      = opt['loss']['lamd']['Rec']    
        lw_Eec      = opt['loss']['lamd']['Eec']   
        lw_Msg      = opt['loss']['lamd']['Msg']
        lw_Con      = opt['loss']['lamd']['Con']

        lamd_ms_Rec = opt['loss']['lamd']['milestones_Rec']      
        lamd_ms_Enc = opt['loss']['lamd']['milestones_Eec']     
        lamd_ms_Msg = opt['loss']['lamd']['milestones_Msg']
        lamd_ms_Con = opt['loss']['lamd']['milestones_Con']
    
        length_rec      = len(lw_Rec)
        length_enc      = len(lw_Eec)
        length_msg      = len(lw_Msg)

        length_Con = len(lw_Con)
        for i in range(length_rec):
            if current_step <= lamd_ms_Rec[i]:
                lwRec = lw_Rec[i]
                break
            elif lamd_ms_Rec[i] < current_step <= lamd_ms_Rec[i + 1]:
                lwRec = lw_Rec[i + 1]
                break
        for i in range(length_enc):
            if current_step <= lamd_ms_Enc[i]:
                lwEnc = lw_Eec[i]
                break
            elif lamd_ms_Enc[i] < current_step <= lamd_ms_Enc[i + 1]:
                lwEnc = lw_Eec[i + 1]
                break
        for i in range(length_msg):
            if current_step <= lamd_ms_Msg[i]:
                lwMsg = lw_Msg[i]
                break
            elif lamd_ms_Msg[i] < current_step <= lamd_ms_Msg[i + 1]:
                lwMsg = lw_Msg[i + 1]
                break
        for i in range(length_Con):
            if current_step <= lamd_ms_Con[i]:
                lwCon = lw_Con[i]
                break
            elif lamd_ms_Con[i] < current_step <= lamd_ms_Con[i + 1]:
                lwCon = lw_Con[i + 1]
                break
   
        lossWeight = {} 
        lossWeight['lwRec'] = lwRec        
        lossWeight['lwEnc'] = lwEnc       
        lossWeight['lwMsg'] = lwMsg       
        lossWeight['lwCon'] = lwCon  

    return lossWeight


def func_loss(lw, loss_RecImg, loss_encoded, loss_RecMsg,loss_Con):
    train_loss = 0
    if lw['lwRec'] != 0 :
        train_loss = train_loss + lw['lwRec'] * loss_RecImg
    if lw['lwEnc'] != 0 :
        train_loss = train_loss + lw['lwEnc'] * loss_encoded
    if lw['lwMsg'] != 0 :
        train_loss = train_loss + lw['lwMsg'] * loss_RecMsg

    if lw['lwCon'] != 0 :
        train_loss = train_loss + lw['lwCon'] * loss_Con

    return train_loss





def func_mean_filter_None(base, input, type= 'add'):
    #
    if type == 'add':
        if base == None or input == None:
            return None
        else:
            return  base+input.item()
    #
    if type == 'div':
        if base == None or input == None:
            return 'None'
        else:
            return base/input


def bitWise_accurary(msg_fake, message, opt):
    #
    if msg_fake == None:
        return None, None
    else:
        #
        if opt['datasets']['msg']['mod_a']:     
            DecodedMsg_rounded = msg_fake.detach().cpu().numpy().round().clip(0, 1)
        elif opt['datasets']['msg']['mod_b']:   
            DecodedMsg_rounded = msg_fake.detach().cpu().numpy().round().clip(-1, 1)
            DecodedMsg_rounded, message = (DecodedMsg_rounded + 1) / 2, (message + 1) / 2 
        #
        diff = DecodedMsg_rounded - message.detach().cpu().numpy()
        count = np.sum(np.abs(diff))
        #
        accuracy = (1 - count / (opt['train']['batch_size'] * opt['network']['message_length'])) 
        BitWise_AvgErr = count / (opt['train']['batch_size'] * opt['network']['message_length'])
        #
        return accuracy * 100, BitWise_AvgErr



def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def filter_None_value(input):
    return 'None' if input == None else input 


def log_info(lw, current_epoch, total_epochs, current_step, Lr_current, psnr_wm2po, psnr_noisy2po, psnr_rec2po, psnr_wm2noisy,
            ssim_wm2po, ssim_rec2po, BitWise_AvgErr, train_loss=None, loss_RecImg=None,
            loss_RecMsg=None, loss_encoded=None, loss_simsiam=None, noise_choice=None):
    
    BitWise_AvgErr = filter_None_value(BitWise_AvgErr)
    
    logging.info(f"Epoch {current_epoch}/{total_epochs} | Step {current_step}")
    logging.info(f"Learning Rate: {Lr_current:.7f} | Noise: {noise_choice}")
    logging.info(f"Loss Weights: Rec={lw['lwRec']} Enc={lw['lwEnc']} Msg={lw['lwMsg']} Con={lw['lwCon']}")
    logging.info(f"Losses: RecMsg={loss_RecMsg.item():.6f} Encoded={loss_encoded.item():.6f} JCL={loss_simsiam.item():.6f}")
    logging.info(f"Quality Metrics:")
    logging.info(f"  PSNR: WM→PO={psnr_wm2po.item():.1f} | Noisy→PO={psnr_noisy2po.item():.1f} | Rec→PO={psnr_rec2po.item():.1f} | WM→Noisy={psnr_wm2noisy.item():.1f}")
    logging.info(f"  SSIM: WM→PO={ssim_wm2po.item():.4f} | Rec→PO={ssim_rec2po.item():.4f}")
    logging.info(f"Bit Error Rate: {BitWise_AvgErr}")
    logging.info('-' * 80)


def log_info_test(current_step, total_steps, Lr_current, psnr_wm2po, psnr_noisy2po, psnr_rec2po, psnr_wm2noisy,
                 ssim_wm2po, ssim_rec2po, BitWise_AvgErr, noise_choice):
    
    BitWise_AvgErr = filter_None_value(BitWise_AvgErr)
    
    logging.info(f"Test Step {current_step}/{total_steps}")
    logging.info(f"Learning Rate: {Lr_current:.7f} | Noise: {noise_choice}")
    logging.info(f"Quality Metrics:")
    logging.info(f"  PSNR: WM→PO={psnr_wm2po.item():.1f} | Noisy→PO={psnr_noisy2po.item():.1f} | Rec→PO={psnr_rec2po.item():.1f} | WM→Noisy={psnr_wm2noisy.item():.1f}")
    logging.info(f"  SSIM: WM→PO={ssim_wm2po.item():.4f} | Rec→PO={ssim_rec2po.item():.4f}")
    logging.info(f"Bit Error Rate: {BitWise_AvgErr}")
    logging.info('-' * 80)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


class ReconstructionImgLoss(nn.Module):
    def __init__(self, opt, eps=1e-6):
        super(ReconstructionImgLoss, self).__init__()
        self.losstype = opt['loss']['type']['TypeRecImg']
        self.eps = eps
        self.N = opt['network']['input']['in_img_nc'] * opt["network"]['H'] * opt["network"]['W']

    def forward(self, true_img, fake_img):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((fake_img - true_img) ** 2, (1, 2, 3))) / self.N
        elif self.losstype == 'l1':
            diff = fake_img - true_img
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3))) / self.N
        else:
            print("reconstruction loss type error!")
            return 0



class ReconstructionMsgLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructionMsgLoss, self).__init__()
        self.losstype = opt['loss']['type']['TyptRecMsg']
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, messages, decoded_messages):
        if self.losstype == 'mse':
            return self.mse_loss(messages, decoded_messages)
        elif self.losstype == 'bce':
            return self.bce_loss(messages, decoded_messages)
        elif self.losstype == 'bce_logits':
            return self.bce_logits_loss(messages, decoded_messages)
        else:
            print("ReconstructionMsgLoss loss type error!")
            return 0


class EncodedLoss(nn.Module):
    def __init__(self, opt, eps=1e-6):
        super(EncodedLoss, self).__init__()
        self.losstype = opt['loss']['type']['TyprEncoded']
        self.eps = eps
        self.N = opt['network']['input']['in_img_nc'] * opt["network"]['H'] * opt["network"]['W']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, true_img, watermarking_img):
        true_img_normalized = true_img * 2 - 1
        watermarking_img_normalized = watermarking_img * 2 - 1
 

        if self.losstype == 'l2':
            loss_mask = 1
            l2_loss = torch.mean(torch.sum((loss_mask * watermarking_img - true_img) ** 2, (1, 2, 3))) / self.N
            return l2_loss 
        elif self.losstype == 'l1':
            diff = watermarking_img - true_img
            l1_loss = torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3))) / self.N
            return l1_loss 
        else:
            print("EncodedLoss loss type error!")
            return 0