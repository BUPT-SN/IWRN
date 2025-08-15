import torch
import torch.nn as nn

import utils.utils as utils
import os
import numpy as np
import logging
import random
from models.IWRN import IWRN
import torch.optim.lr_scheduler as lr
import utils.checkpoint as check
import kornia
import torch.nn.functional as F
from models.JCL import JCLE, JCLD
import time
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, opt, device, path_in):
        super(Network, self).__init__()
        self.opt = opt
        self.device = device
        self.device_ids = opt['train']['device_ids']
        self.IWRN = IWRN(opt, self.device)
        self.IWRN = nn.DataParallel(self.IWRN, device_ids=self.device_ids)
        self.Checkpoint = check.Checkpoint(path_in['path_checkpoint'], opt)
        if opt['train']['resume']['Empty'] != True:
            self.resume()
        self.JCLE = JCLE().to(self.device)
        self.JCLIWRN = JCLD(self.IWRN.module)
        self.JCLIWRN= torch.nn.DataParallel(self.JCLIWRN, device_ids=self.device_ids)
        self.RecImgLoss = utils.ReconstructionImgLoss(opt)
        self.RecMsgLoss = utils.ReconstructionMsgLoss(opt)
        self.encodedLoss = utils.EncodedLoss(opt)
        self.Lr_current = opt['lr']['start_lr']

        if opt['lr']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.JCLIWRN.parameters()),
                                              lr=self.Lr_current)
        elif opt['lr']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.JCLIWRN.parameters()),
                                             lr=self.Lr_current, momentum=0.9)
        self.lr_milestones = opt['lr']['milestones']
        self.lr_gamma = opt['lr']['gamma']
        self.scheduler = lr.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        self.current_step = 0

        self.lw = utils.loss_lamd(self.current_step, opt)
        self.img_w_folder_tra = path_in['img_w_folder_tra']
        self.loss_w_folder = path_in['loss_w_folder']
        self.log_folder = path_in['log_folder']
        self.img_w_folder_val = path_in['img_w_folder_val']
        self.img_w_folder_test = path_in['img_w_folder_test']
        self.time_now_NewExperiment = path_in['time_now_NewExperiment']



    def train(self, train_data, current_epoch):
        logging.info('--------------------------------------------------------\n')
        logging.info('##### train #####\n')
        self.current_epoch = current_epoch
        
        #
        with torch.enable_grad():
            #
            loss_per_epoch_sum = 0
            loss_per_epoch_msg = 0
            loss_per_epoch_enc = 0
            train_step = 0
            for _, image in enumerate(train_data):
                self.IWRN.train()
                self.optimizer.zero_grad()
                image = image.to(self.device)
                if self.opt['datasets']['msg']['mod_a']:  # msg in [0, 1]
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                elif self.opt['datasets']['msg']['mod_b']:  # msg in [-1, 1]
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()
                if self.opt['noise']['option'] == 'Combined':
                    if current_epoch <= 40:
                        noise_choice = 'Space'
                    else:
                        NoiseName = self.opt['noise']['Combined']['names']
                        noise_choice = random.choice(NoiseName)
                else:
                    noise_choice = self.opt['noise']['option']
                z0, p0, watermarking_img, noised_img, img_fake, msg_fake, _, residual1, residual2 = self.JCLIWRN(
                    image, message, noise_choice, True)
                z1, p1, _, _, _, _, _, _, _ = self.JCLIWRN(watermarking_img, message, noise_choice, True)

                loss_RecImg = self.RecImgLoss(image, img_fake)
                loss_RecMsg = utils.func_loss_RecMsg(self.RecMsgLoss, message, msg_fake)
                loss_encoded = self.encodedLoss(image, watermarking_img)
                z2, p2 = self.JCLE(residual1)
                z3, p3= self.JCLE(residual2)
                criterion_ssl = NegativeCosineSimilarity()
                loss_simsiam = 0.0005 * (criterion_ssl(z0, p1) + criterion_ssl(z1, p0)) + 0.0005 * (criterion_ssl(z2, p3) + criterion_ssl(z3, p2))
                train_loss = utils.func_loss(self.lw, loss_RecImg, loss_encoded, loss_RecMsg, loss_simsiam)
                train_loss.backward()
                self.optimizer.step()

                # log print
                if self.current_step % self.opt['train']['logs_per_step'] == 0:
                    psnr_wm2po, psnr_no2po, psnr_rec2po, psnr_wm2no, ssim_wm2po, ssim_rec2po, acc, BitWise_AvgErr1, \
                    = self.psnr_ssim_acc(image, watermarking_img, noised_img, img_fake,
                                                               msg_fake, message)
                    #
                    utils.log_info(self.lw, self.current_epoch, self.opt['train']['epoch'], self.current_step,
                                   self.Lr_current, psnr_wm2po, psnr_no2po, psnr_rec2po, psnr_wm2no, \
                                   ssim_wm2po, ssim_rec2po, BitWise_AvgErr1, train_loss, loss_RecImg,
                                   loss_RecMsg, loss_encoded,loss_simsiam, noise_choice)

                # save images
                if self.current_step % self.opt["train"]['saveTrainImgs_per_step'] == 0:
                    utils.mkdir(self.img_w_folder_tra)
                    utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch,
                                      self.current_step, self.img_w_folder_tra, self.time_now_NewExperiment, self.opt,
                                      resize_to=None)

                # break code if 'nan'
                if torch.isnan(train_loss):
                    logging.info("Invalid loss <nan>, break code !")
                    exit()

                # step update
                self.current_step += 1
                train_step += 1
                loss_per_epoch_sum = loss_per_epoch_sum + (loss_RecMsg.item() + loss_encoded.item())
                loss_per_epoch_msg = loss_per_epoch_msg + loss_RecMsg.item()
                loss_per_epoch_enc = loss_per_epoch_enc + loss_encoded.item()

                # loss weight update
                self.lw = utils.loss_lamd(self.current_step, self.opt)

                # lr update
                self.scheduler.step()
                self.Lr_current = self.scheduler.get_last_lr()[0]

            # Checkpoint
            if self.current_epoch % self.opt['train']['checkpoint_per_epoch'] == 0:
                logging.info('Checkpoint: Saving cinNets and training states.')
                self.Checkpoint.save(self.IWRN, self.current_step, self.current_epoch, 'cinNet')

            # write losses
            utils.mkdir(self.loss_w_folder)
            utils.write_losses(os.path.join(self.loss_w_folder, 'train-{}.txt'.format(self.time_now_NewExperiment)),
                               self.current_epoch, loss_per_epoch_sum / train_step, loss_per_epoch_msg / train_step,
                               loss_per_epoch_enc / train_step)

    def validation(self, val_data, current_epoch):
        
        logging.info('--------------------------------------------------------\n')
        logging.info('##### validation #####\n')
        self.current_epoch = current_epoch
    
        val_step = 0
        psnr_wm2no_mean = 0
        psnr_wm2po_mean = 0
        psnr_rec2po_mean = 0
        psnr_no2po_mean = 0
        BitWise_AvgErr1_mean = 0

        ssim_wm2po_mean = 0
        ssim_rec2po_mean = 0

        loss_per_val_sum = 0
        loss_per_val_msg = 0
        loss_per_val_enc = 0
        with torch.no_grad():
            for _, image in enumerate(val_data):  
                self.IWRN.eval()
                image = image.to(self.device)
                if self.opt['datasets']['msg']['mod_a']:  
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                elif self.opt['datasets']['msg']['mod_b']: 
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()

                if self.opt['noise']['option'] == 'Combined':
                    NoiseName = self.opt['noise']['Combined']['names']
                    noise_choice = random.choice(NoiseName)
                else:
                    noise_choice = self.opt['noise']['option']
                
                z0, p0, watermarking_img, noised_img, img_fake, msg_fake, _, residual1, residual2 = self.JCLD(
                    image, message, noise_choice, True)
                z1, p1, _, _, _, _, _, _, _ = self.JCLD(watermarking_img, message, noise_choice, True)
                z2, p2 = self.JCLE(residual1)
                z3, p3 = self.JCLE(residual2)
                criterion_ssl = NegativeCosineSimilarity()
                loss_simsiam = 0.005 * (criterion_ssl(z0, p1) + criterion_ssl(z1, p0)) + 0.005 * (criterion_ssl(z2, p3) + criterion_ssl(z3, p2))
  
                loss_RecImg = self.RecImgLoss(image, img_fake)
                loss_RecMsg = utils.func_loss_RecMsg(self.RecMsgLoss, message, msg_fake)
                loss_encoded = self.encodedLoss(image, watermarking_img)
                val_loss = loss_RecMsg + loss_encoded

                if val_step % self.opt['train']['val']['logs_per_step'] == 0:
                    psnr_wm2po, psnr_no2po, psnr_rec2po, psnr_wm2no, ssim_wm2po, ssim_rec2po, acc, BitWise_AvgErr1, \
                   = self.psnr_ssim_acc(image, watermarking_img, noised_img, img_fake,
                                                               msg_fake, message)
                    
                    utils.log_info(self.lw, self.current_epoch, 0, self.current_step, self.Lr_current, psnr_wm2po,
                                   psnr_no2po, psnr_rec2po, psnr_wm2no, \
                                   ssim_wm2po, ssim_rec2po, BitWise_AvgErr1, val_loss, loss_RecImg,
                                   loss_RecMsg, loss_encoded,loss_simsiam, noise_choice)
                    

                if val_step == self.opt["train"]['saveValImgs_in_step']:
                    utils.mkdir(self.img_w_folder_val)
                    utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch,
                                      self.current_step, self.img_w_folder_val, self.time_now_NewExperiment, self.opt,
                                      resize_to=None)

                val_step += 1
                loss_per_val_sum = loss_per_val_sum + (loss_RecMsg.item() + loss_encoded.item())
                loss_per_val_msg = loss_per_val_msg + loss_RecMsg.item()
                loss_per_val_enc = loss_per_val_enc + loss_encoded.item()

                psnr_wm2no_mean = psnr_wm2no_mean + psnr_wm2no.item()
                psnr_wm2po_mean = psnr_wm2po_mean + psnr_wm2po.item()
                psnr_rec2po_mean = psnr_rec2po_mean + psnr_rec2po.item()
                psnr_no2po_mean = psnr_no2po_mean + psnr_no2po.item()
                
                ssim_wm2po_mean = ssim_wm2po_mean + ssim_wm2po.item()
                ssim_rec2po_mean = ssim_rec2po_mean + ssim_rec2po.item()

                BitWise_AvgErr_mean = utils.func_mean_filter_None(BitWise_AvgErr1_mean, BitWise_AvgErr1, 'add')
                

            print_BitWise_AvgErr_mean = utils.func_mean_filter_None(BitWise_AvgErr_mean, val_step, 'div')
            

            logging.info(f"Quality Metrics:")
            logging.info(f"  PSNR: WM→PO={psnr_wm2po_mean / val_step:.1f} | Noisy→PO={psnr_no2po_mean / val_step:.1f} | Rec→PO={psnr_rec2po_mean / val_step:.1f} | WM→Noisy={psnr_wm2no_mean / val_step:.1f}")
            logging.info(f"  SSIM: WM→PO={ssim_wm2po_mean / val_step:.4f} | Rec→PO={ssim_rec2po_mean / val_step:.4f}")
            logging.info(f"Bit Error Rate: {print_BitWise_AvgErr_mean}")
            logging.info('-' * 80)

            #
            utils.mkdir(self.loss_w_folder)
            utils.write_losses(os.path.join(self.loss_w_folder, 'val-{}.txt'.format(self.time_now_NewExperiment)),
                               self.current_epoch, loss_per_val_sum / val_step, loss_per_val_msg / val_step,
                               loss_per_val_enc / val_step)
        #
        logging.info('--------------------------------------------------------\n')

    def test(self, test_data, current_epoch):
        logging.info('--------------------------------------------------------\n')
        logging.info('##### test only #####\n')
        self.current_epoch = current_epoch
        
        with torch.no_grad():
            self.IWRN.eval()
            
            test_step = 0
            total_steps = len(test_data)
            
            psnr_wm2no_mean = 0
            psnr_wm2po_mean = 0
            psnr_rec2po_mean = 0
            psnr_no2po_mean = 0
            BER_mean = 0
            ssim_wm2po_mean = 0
            ssim_rec2po_mean = 0
            
            for _, image in enumerate(test_data):
                start_time = time.time()  
                
                if (test_step * self.opt['train']['batch_size']) >= self.opt['datasets']['test']['num']:
                    break
                
                image = image.to(self.device)
                
                if self.opt['datasets']['msg']['mod_a']: 
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                elif self.opt['datasets']['msg']['mod_b']: 
                    message = torch.Tensor(
                        np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(
                        self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()
                
 
                if self.opt['noise']['option'] == 'Combined':
                    NoiseName = self.opt['noise']['Combined']['names']
                    noise_choice = random.choice(NoiseName)
                else:
                    noise_choice = self.opt['noise']['option']
                
                _, _, watermarking_img, noised_img, img_fake, msg_fake, msg_nsm, _, _ = \
                    self.IWRN(image, message, noise_choice, False)
                
                psnr_wm2po, psnr_no2po, psnr_rec2po, psnr_wm2no, ssim_wm2po, ssim_rec2po, acc, BitWise_AvgErr1\
                 = self.psnr_ssim_acc(image, watermarking_img, noised_img, img_fake, msg_fake, message)
                
                psnr_wm2no_mean += psnr_wm2no 
                psnr_wm2po_mean += psnr_wm2po
                psnr_rec2po_mean += psnr_rec2po
                psnr_no2po_mean += psnr_no2po
                
                ssim_wm2po_mean += ssim_wm2po.item() 
                ssim_rec2po_mean += ssim_rec2po.item()
                
                _, ber = utils.bitWise_accurary(msg_nsm, message, self.opt)
                BER_mean += ber
                
                if test_step % self.opt["train"]['logTest_per_step'] == 0:
                    utils.log_info_test(test_step, total_steps, self.Lr_current, psnr_wm2po, psnr_no2po, psnr_rec2po,
                                        psnr_wm2no, ssim_wm2po, ssim_rec2po, BitWise_AvgErr1, noise_choice)
                
                if test_step % self.opt["train"]['saveTestImgs_per_step'] == 0:
                    utils.mkdir(self.img_w_folder_test)
                    utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch, test_step,
                                    self.img_w_folder_test, self.time_now_NewExperiment, self.opt, resize_to=None)
                
                end_time = time.time()  
                elapsed_time = end_time - start_time
                print(f"Step {test_step} took {elapsed_time:.4f} seconds.")
                
                test_step += 1

            logging.info(f"Quality Metrics:")
            logging.info(f"  PSNR: WM→PO={psnr_wm2po_mean.item()/test_step}")
            logging.info(f"  PSNR: Noisy→PO={psnr_no2po_mean.item()/test_step}")
            logging.info(f"  PSNR: Rec→PO={psnr_rec2po_mean.item()/test_step}")
            logging.info(f"  PSNR: WM→Noisy={psnr_wm2no_mean.item()/test_step}")
            logging.info(f"  SSIM: WM→PO={ssim_wm2po_mean/test_step}")
            logging.info(f"  SSIM: Rec→PO={ssim_rec2po_mean/test_step}")
            logging.info(f"Bit Error Rate: {BER_mean/test_step}")
            logging.info(f"Accuracy: {(1 - BER_mean / test_step) * 100}%")
        logging.info('-' * 80)
        logging.info('Test end !')

    def psnr_ssim_acc(self, image, watermarking_img, noised_img, img_fake, msg_fake, message):
        # psnr
        psnr_wm2po = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((watermarking_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_no2po = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((noised_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_rec2po = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((img_fake.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_wm2no = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((noised_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        # ssim
        ssim_wm2po = kornia.metrics.ssim(
            ((image + 1) / 2).clamp(0, 1),
            ((watermarking_img.detach() + 1) / 2).clamp(0, 1),
            window_size=19,
        ).mean()
        ssim_rec2po = kornia.metrics.ssim(
            ((image + 1) / 2).clamp(0, 1),
            ((img_fake.detach() + 1) / 2).clamp(0, 1),
            window_size=19,
        ).mean()
        # acc
        acc, BitWise_AvgErr = utils.bitWise_accurary(msg_fake, message, self.opt)
        
        return psnr_wm2po, psnr_no2po, psnr_rec2po, psnr_wm2no, ssim_wm2po, ssim_rec2po, acc, BitWise_AvgErr

    def resume(self):
        device_id = torch.cuda.current_device()
        if self.opt['train']['resume']['one_pth']:
            resume_state = torch.load(self.opt['path']['resume_state_1pth'],
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            self.IWRN = self.Checkpoint.resume_training(self.IWRN, 'IWRN_Net', resume_state)
        else:
            resume_state = torch.load(self.opt['path']['resume_state_IWRN_Net'],
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            self.IWRN = self.Checkpoint.resume_training(self.IWRN, 'model_state_dict', resume_state)
            resume_state_nsmNet = torch.load(self.opt['path']['resume_state_nsmNet'],
                                             map_location=lambda storage, loc: storage.cuda(device_id))
            self.IWRN.module.nsm_model = self.Checkpoint.resume_training(self.IWRN.module.nsm_model, 'nsmNet',
                                                                           resume_state_nsmNet)


class NegativeCosineSimilarity(nn.Module):
    def __init__(self):
        super(NegativeCosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        return -F.cosine_similarity(x1, x2).mean()


