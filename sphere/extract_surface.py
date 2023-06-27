import sys
import os
os.chdir('..')
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import nibabel as nib
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from surface.net import MeshDeform
from utils.mesh import (
    apply_affine_mat,
    cot_laplacian_smooth)


def extract_surface(args):
    # load argumentations
    device = args.device
    surf_hemi = args.surf_hemi
    data_split = args.data_split

    # load input surface template
    affine_temp = nib.load(
        './template/dhcp_week-40_template_T2w.nii.gz').affine
    surf_temp = nib.load(
        './template/dhcp_week-40_hemi-'+surf_hemi+'_init.surf.gii')
    vert_temp = surf_temp.agg_data('pointset')
    face_temp = surf_temp.agg_data('triangle')
    vert_temp = apply_affine_mat(
        vert_temp, np.linalg.inv(affine_temp))
    face_temp = face_temp[:,[2,1,0]]
    if surf_hemi == 'left':
        vert_temp[:,0] = vert_temp[:,0] - 64
    vert_in = torch.Tensor(vert_temp[None]).to(device)
    face_in = torch.LongTensor(face_temp[None]).to(device)

    # load nn model
    model_dir = './surface/model/model_hemi-'+surf_hemi+'_wm.pt'
    surf_recon = MeshDeform(
        C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0,
        interpolation='tricubic', device=device)
    surf_recon.load_state_dict(
        torch.load(model_dir, map_location=device))
    
    # directory of data
    subj_dir = './surface/data/'+data_split+'/'
    save_dir = './sphere/data/'+data_split+'/'
    subj_list = sorted(glob.glob(subj_dir+'/*'))

    for i in tqdm(range(len(subj_list))):
        subj_dir = subj_list[i]
        subj_id = subj_list[i].split('/')[-1]

        # load input volume
        img_in = nib.load(
            subj_dir+'/'+subj_id+'_T2w_affine.nii.gz')
        vol_in = img_in.get_fdata()
        affine_in = img_in.affine
        vol_in = (vol_in / vol_in.max()).astype(np.float32)

        # clip left/right hemisphere
        if surf_hemi == 'left':
            vol_in = vol_in[None, 64:]
        elif surf_hemi == 'right':
            vol_in = vol_in[None, :112]
        vol_in = torch.Tensor(vol_in[None]).to(device)

        # extract wm surface
        with torch.no_grad():
            vert_wm = surf_recon(vert_in, vol_in)
            vert_wm = cot_laplacian_smooth(vert_wm, face_in, n_iters=1)

        # transform to original space
        vert_wm = vert_wm[0].cpu().numpy()
        face_wm = face_in[0].cpu().numpy()
        if surf_hemi == 'left':
            vert_wm = vert_wm + [64,0,0]
        vert_wm = apply_affine_mat(vert_wm, affine_in)
        face_wm = face_wm[:,[2,1,0]]

        # save surface mesh as freesurfer file
        if not os.path.exists(save_dir+subj_id):
            os.makedirs(save_dir+subj_id)
        if surf_hemi == 'left':
            hemi = 'lh'
        elif surf_hemi == 'right':
            hemi = 'rh'
        nib.freesurfer.write_geometry(
            save_dir+subj_id+'/'+hemi+'.white',vert_wm, face_wm)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Surface Recon")
    
    parser.add_argument('--data_split', default='train', type=str, help="[train, valid, test]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")

    args = parser.parse_args()
    extract_surface(args)
