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

from sphere.net.sunet import SphereDeform
from sphere.net.loss import(
    edge_distortion,
    area_distortion,
    geodesic_distortion)
from utils.mesh import face_area


class SphereDataset(Dataset):
    """
    Dataset class for surface reconstruction
    """
    def __init__(self, args, data_split='train'):
        super(SphereDataset, self).__init__()
        
        # ------ load arguments ------ 
        surf_hemi = args.surf_hemi
        device = args.device

        if surf_hemi == 'left':
            surf_hemi_ = 'lh'
        elif surf_hemi == 'right':
            surf_hemi_ = 'rh'
            
        # load pre-computed barycentric coordinates
        # for sphere interpolation
        barycentric = nib.load('./template/dhcp_week-40_hemi-'+surf_hemi+'_barycentric.gii')
        bc_coord = barycentric.agg_data('pointset')
        face_id = barycentric.agg_data('triangle')

        data_dir = './sphere/data/'+data_split+'/'
        subj_list = sorted(glob.glob(data_dir+'*/'+surf_hemi_+'.white'))
        self.data_list = []
        
        for i in tqdm(range(len(subj_list[:]))):
            subj_dir = '/'.join(subj_list[i].split('/')[:-1])+'/'

            # load input wm surface
            vert_wm_in = nib.freesurfer.read_geometry(
                subj_dir+surf_hemi_+'.white')[0].astype(np.float32)
            # barycentric interpoalataionn: resample to 160k template
            vert_wm_160k = (vert_wm_in[face_id] * bc_coord[...,None]).sum(-2)
            # load gt sphere
            vert_sphere_gt = nib.freesurfer.read_geometry(
                subj_dir+surf_hemi_+'.sphere')[0].astype(np.float32)
            vert_sphere_gt = vert_sphere_gt / 100.
            sphere_data = (vert_wm_in, vert_wm_160k, vert_sphere_gt)
            self.data_list.append(sphere_data)  # add to data list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        surf_data = self.data_list[i]
        return surf_data



def train_loop(args):
    # ------ load arguments ------ 
    surf_hemi = args.surf_hemi  # left or right
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    w_geod = args.w_geod  # weight for nc loss
    w_edge = args.w_edge  # weight for edge loss
    w_area = args.w_area  # weight for edge loss
    
    # start training logging
    logging.basicConfig(
        filename='./sphere/ckpts/log_hemi-'+surf_hemi+'_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SphereDataset(args, data_split='train')
    validset = SphereDataset(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    

    # ------ load input sphere ------
    sphere_in = nib.load(
        './template/dhcp_week-40_hemi-'+surf_hemi+'_sphere.surf.gii')
    vert_sphere_in = sphere_in.agg_data('pointset')
    face_in = sphere_in.agg_data('triangle')
    vert_sphere_in = torch.Tensor(vert_sphere_in[None]).to(device)
    face_in = torch.LongTensor(face_in[None]).to(device)
    edge_in = torch.cat([face_in[0,:,[0,1]],
                         face_in[0,:,[1,2]],
                         face_in[0,:,[2,0]]], dim=0).T

    # ------ load template sphere (160k) ------
    sphere_160k = nib.load('./template/sphere_163842.surf.gii')
    vert_sphere_160k = sphere_160k.agg_data('pointset')
    face_160k = sphere_160k.agg_data('triangle')
    vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
    face_160k = torch.LongTensor(face_160k[None]).to(device)
    
    # ------ load model ------
    sphere_proj = SphereDeform(
        C_in=6, C_hid=[32, 64, 128, 256, 256], device=device)
    optimizer = optim.Adam(sphere_proj.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            vert_wm_in, vert_wm_160k, vert_sphere_gt = data
            vert_wm_in = vert_wm_in.to(device).float()
            vert_wm_160k = vert_wm_160k.to(device).float()
            vert_sphere_gt = vert_sphere_gt.to(device).float()
            feat_160k = torch.cat([vert_sphere_160k, vert_wm_160k], dim=-1)

            optimizer.zero_grad()
            vert_sphere_pred = sphere_proj(
                feat_160k, vert_sphere_in, n_steps=7)

            # supervised geodesic loss
            geod_loss = geodesic_distortion(
                vert_sphere_pred, vert_sphere_gt, n_samples=500000)
            # unsupervised metric distortion loss
            edge_loss = edge_distortion(
                vert_sphere_pred, vert_wm_in, edge_in)
            area_loss = area_distortion(
                vert_sphere_pred, vert_wm_in, face_in)
            loss = w_geod*geod_loss + w_edge*edge_loss + w_area*area_loss

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                geod_error = []
                edge_error = []
                area_error = []
                for idx, data in enumerate(validloader):
                    vert_wm_in, vert_wm_160k, vert_sphere_gt = data
                    vert_wm_in = vert_wm_in.to(device).float()
                    vert_wm_160k = vert_wm_160k.to(device).float()
                    vert_sphere_gt = vert_sphere_gt.to(device).float()
                    feat_160k = torch.cat([vert_sphere_160k, vert_wm_160k], dim=-1)

                    vert_sphere_pred = sphere_proj(
                        feat_160k, vert_sphere_in, n_steps=7)

                    # supervised geodesic loss
                    geod_loss = geodesic_distortion(
                        vert_sphere_pred, vert_sphere_gt, n_samples=500000)
                    # unsupervised metric distortion loss
                    edge_loss = edge_distortion(
                        vert_sphere_pred, vert_wm_in, edge_in)
                    area_loss = area_distortion(
                        vert_sphere_pred, vert_wm_in, face_in)
                    
                    geod_error.append(geod_loss.item())
                    edge_error.append(edge_loss.item())
                    area_error.append(area_loss.item())
                    
            logging.info('epoch:{}'.format(epoch))
            logging.info('geodesic error:{}'.format(np.mean(geod_error)))
            logging.info('edge error:{}'.format(np.mean(edge_error)))
            logging.info('area error:{}'.format(np.mean(area_error)))
            logging.info('-------------------------------------')

            # save model checkpoints
            torch.save(sphere_proj.state_dict(),
                       './sphere/ckpts/model_hemi-'+surf_hemi+'_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Spherical Mapping")
    
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")

    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")

    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=100, type=int, help="number of training epochs")

    parser.add_argument('--w_geod', default=1.0, type=float, help="weight for geodesic distortion loss")
    parser.add_argument('--w_edge', default=1.0, type=float, help="weight for edge distortion loss")
    parser.add_argument('--w_area', default=0.5, type=float, help="weight for area distortion loss")


    args = parser.parse_args()
    
    train_loop(args)