import sys
import os
os.chdir('..')
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import nibabel as nib
import glob
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from seg.unet import UNet


class SegDataset(Dataset):
    """
    Dataset class for brain mask extration
    """
    def __init__(self, data_split='train'):
        super(Dataset, self).__init__()
        
        # ------ load path ------ 
        data_dir = './seg/data/'+data_split+'/*'
        subj_list = sorted(glob.glob(data_dir))
        self.data_list = []
        
        for i in tqdm(range(len(subj_list))):
            subj_dir = subj_list[i]
            subj_id = subj_list[i].split('/')[-1]
            
            # ------ load input volume ------
            img_t2 = nib.load(subj_dir+'/'+subj_id+'_T2w_affine.nii.gz')
            vol_in = img_t2.get_fdata()
            vol_in = (vol_in[None] / vol_in.max()).astype(np.float32)
            
            ribbon_gt = nib.load(subj_dir+'/'+subj_id+'_ribbon.nii.gz')
            ribbon_gt = ribbon_gt.get_fdata()[None].astype(bool)
            
            seg_data = (vol_in, ribbon_gt)
            self.data_list.append(seg_data)  # add to data list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        surf_data = self.data_list[i]
        return surf_data

    

def train_loop(args):
    # ------ load arguments ------ 
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    
    # start training logging
    logging.basicConfig(
        filename='./seg/ckpts/log_seg_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SegDataset(data_split='train')
    validset = SegDataset(data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    # ------ initialize model ------ 
    logging.info("initalize model ...")
    seg_ribbon = UNet(C_in=1, C_hid=[16,32,64,128,128], C_out=1).to(device)
    optimizer = optim.Adam(seg_ribbon.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            vol_in, ribbon_gt = data
            vol_in = vol_in.to(device).float()
            ribbon_gt = ribbon_gt.to(device).float()
            
            ribbon_pred = torch.sigmoid(seg_ribbon(vol_in))
            
            loss = nn.BCELoss()(ribbon_pred, ribbon_gt)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                valid_error = []
                valid_dice = []
                for idx, data in enumerate(validloader):
                    optimizer.zero_grad()
                    vol_in, ribbon_gt = data
                    vol_in = vol_in.to(device).float()
                    ribbon_gt = ribbon_gt.to(device).float()
                    ribbon_pred = torch.sigmoid(seg_ribbon(vol_in))
                    error = nn.BCELoss()(ribbon_pred, ribbon_gt).item()
                    valid_error.append(error)
                    
                    # compute dice
                    ribbon_gt = ribbon_gt.cpu().numpy()
                    ribbon_pred = (ribbon_pred>0.5).float().cpu().numpy()
                    dice = 2*(ribbon_pred * ribbon_gt).sum() / \
                           (ribbon_pred.sum() + ribbon_gt.sum() + 1e-12)
                    valid_dice.append(dice)
                    
                logging.info('epoch:{}'.format(epoch))
                logging.info('valid error:{}'.format(np.mean(valid_error)))
                logging.info('dice score:{}'.format(np.mean(valid_dice)))
                logging.info('-------------------------------------')
                
            # save model checkpoints
            torch.save(seg_ribbon.state_dict(),
                       './seg/ckpts/model_seg_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cortical Ribbon")
    parser.add_argument('--device', default="cuda", type=str, help="cuda or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")

    args = parser.parse_args()
    
    train_loop(args)
