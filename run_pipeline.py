import os
import glob
import time
import argparse
import subprocess
import numpy as np
import ants
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.unet import UNet
from surface.net import MeshDeform
from sphere.net.sunet import SphereDeform
from sphere.net.loss import (
    edge_distortion,
    area_distortion)

from utils.mesh import (
    apply_affine_mat,
    cot_laplacian_smooth,
    laplacian_smooth)

from utils.register import (
    registration,
    ants_trans_to_mat)

from utils.io import (
    save_numpy_to_nifti,
    save_gifti_surface,
    save_gifti_metric,
    create_wb_spec)

from utils.inflate import (
    generate_inflated_surfaces,
    wb_generate_inflated_surfaces)

from utils.metric import (
    metric_dilation,
    cortical_thickness,
    curvature,
    sulcal_depth,
    myelin_map)



# ------ load hyperparameters ------ 
parser = argparse.ArgumentParser(description="dHCP DL Surface Pipeline")
parser.add_argument('--in_dir', default='./in_dir/', type=str,
                    help='Diectory containing input images.')
parser.add_argument('--out_dir', default='./out_dir/', type=str,
                    help='Directory for saving the output of the pipeline.')
parser.add_argument('--T2', default='_T2w.nii.gz', type=str,
                    help='Suffix of T2 image file.')
parser.add_argument('--T1', default='_T1w.nii.gz', type=str,
                    help='Suffix of T1 image file.')
parser.add_argument('--mask', default='_brainmask.nii.gz', type=str,
                    help='Suffix of brain mask file.')
parser.add_argument('--device', default='cuda', type=str,
                    help='Device for running the pipeline: [cuda, cpu]')
parser.add_argument('--verbose', action='store_true',
                    help='Print debugging information.')
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
t2_suffix = args.T2
t1_suffix = args.T1
mask_suffix = args.mask
device = args.device
verbose = args.verbose
max_regist_iter = 5
min_regist_dice = 0.9


# ------ load nn model ------ 
# ribbon segmentation
seg_ribbon = UNet(
    C_in=1, C_hid=[16,32,64,128,128], C_out=1).to(device)
seg_ribbon.load_state_dict(
    torch.load('./seg/model/model_seg.pt',
               map_location=device))

# surface reconstruction
surf_recon_left_wm = MeshDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0,
    interpolation='tricubic', device=device)
surf_recon_right_wm = MeshDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0,
    interpolation='tricubic', device=device)
surf_recon_left_pial = MeshDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0,
    interpolation='tricubic',device=device)
surf_recon_right_pial = MeshDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0,
    interpolation='tricubic',device=device)

surf_recon_left_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-left_wm.pt',
               map_location=device))
surf_recon_right_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-right_wm.pt',
               map_location=device))
surf_recon_left_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-left_pial.pt',
               map_location=device))
surf_recon_right_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-right_pial.pt',
               map_location=device))

# spherical projection
sphere_proj_left = SphereDeform(
    C_in=6, C_hid=[32, 64, 128, 256, 256], device=device)
sphere_proj_right = SphereDeform(
    C_in=6, C_hid=[32, 64, 128, 256, 256], device=device)

sphere_proj_left.load_state_dict(
    torch.load('./sphere/model/model_hemi-left_sphere.pt',
               map_location=device))
sphere_proj_right.load_state_dict(
    torch.load('./sphere/model/model_hemi-right_sphere.pt',
               map_location=device))


# ------ load image atlas ------
img_t2_atlas_ants = ants.image_read(
    './template/dhcp_week-40_template_T2w.nii.gz')
# both ants->nibabel and nibabel->ants need to reload the nifiti file
# so here simply load the image again
affine_t2_atlas = nib.load(
    './template/dhcp_week-40_template_T2w.nii.gz').affine


# ------ load input surface ------
surf_left_in = nib.load(
    './template/dhcp_week-40_hemi-left_init.surf.gii')
vert_left_in = surf_left_in.agg_data('pointset')
face_left_in = surf_left_in.agg_data('triangle')
vert_left_in = apply_affine_mat(
    vert_left_in, np.linalg.inv(affine_t2_atlas))
vert_left_in = vert_left_in - [64,0,0]
face_left_in = face_left_in[:,[2,1,0]]
vert_left_in = torch.Tensor(vert_left_in[None]).to(device)
face_left_in = torch.LongTensor(face_left_in[None]).to(device)

surf_right_in = nib.load(
    './template/dhcp_week-40_hemi-right_init.surf.gii')
vert_right_in = surf_right_in.agg_data('pointset')
face_right_in = surf_right_in.agg_data('triangle')
vert_right_in = apply_affine_mat(
    vert_right_in, np.linalg.inv(affine_t2_atlas))
face_right_in = face_right_in[:,[2,1,0]]
vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
face_right_in = torch.LongTensor(face_right_in[None]).to(device)


# ------ load input sphere ------
sphere_left_in = nib.load(
    './template/dhcp_week-40_hemi-left_sphere.surf.gii')
vert_sphere_left_in = sphere_left_in.agg_data('pointset')
vert_sphere_left_in = torch.Tensor(vert_sphere_left_in[None]).to(device)

sphere_right_in = nib.load(
    './template/dhcp_week-40_hemi-right_sphere.surf.gii')
vert_sphere_right_in = sphere_right_in.agg_data('pointset')
vert_sphere_right_in = torch.Tensor(vert_sphere_right_in[None]).to(device)


# ------ load template sphere (160k) ------
sphere_160k = nib.load('./template/sphere_163842.surf.gii')
vert_sphere_160k = sphere_160k.agg_data('pointset')
face_160k = sphere_160k.agg_data('triangle')
vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
face_160k = torch.LongTensor(face_160k[None]).to(device)


# ------ load pre-computed barycentric coordinates ------
# for sphere interpolation
barycentric_left = nib.load('./template/dhcp_week-40_hemi-left_barycentric.gii')
bc_coord_left = barycentric_left.agg_data('pointset')
face_left_id = barycentric_left.agg_data('triangle')

barycentric_right = nib.load('./template/dhcp_week-40_hemi-right_barycentric.gii')
bc_coord_right = barycentric_right.agg_data('pointset')
face_right_id = barycentric_right.agg_data('triangle')


# ------ run dHCP DL-based surface pipeline ------
if __name__ == '__main__':
    subj_list = sorted(glob.glob(in_dir+'*'+t2_suffix))
    for subj_t2_dir in tqdm(subj_list):
        t_start = time.time()
        subj_id = subj_t2_dir.split('/')[-1][:-len(t2_suffix)]
        subj_in_dir = '/'.join(subj_t2_dir.split('/')[:-1])+'/'
        subj_t1_dir = ''
        subj_mask_dir = ''
        if t1_suffix:
            subj_t1_dir = subj_in_dir + subj_id + t1_suffix
        if mask_suffix:
            subj_mask_dir = subj_in_dir + subj_id + mask_suffix
        t1_exists = False
        mask_exists = False
        if os.path.exists(subj_t1_dir):
            t1_exists = True
        if os.path.exists(subj_mask_dir):
            mask_exists = True

        print('====================')
        print('Start processing subject: {}'.format(subj_id))
        # directory for saving output: out_dir/subj_id/
        subj_out_dir = out_dir + subj_id + '/'
        # create output directory
        if not os.path.exists(subj_out_dir):
            os.mkdir(subj_out_dir)
            # add subject id as prefix
        subj_out_dir = subj_out_dir + subj_id

        # ------ Load Data ------
        print('Load T2 image ...', end=' ')
        # copy T2 image to output directory
        subprocess.run(
            'cp '+subj_t2_dir+' '+subj_out_dir+'_T2w.nii.gz',
            shell=True)

        # load T2 image
        img_t2_orig_ants = ants.image_read(subj_t2_dir)
        img_t2_orig = img_t2_orig_ants.numpy()

        # ants image produces inaccurate affine matrix
        # reload the nifti file to get the affine matrix
        img_t2_orig_nib = nib.load(subj_t2_dir)
        affine_t2_orig = img_t2_orig_nib.affine

        # args for converting numpy.array to ants image
        args_t2_orig_ants = (
            img_t2_orig_ants.origin,
            img_t2_orig_ants.spacing,
            img_t2_orig_ants.direction)
        print('Done.')

        # load brain mask if exists
        if mask_exists:
            # copy brain mask to output directory
            subprocess.run(
                'cp '+subj_mask_dir+' '+subj_out_dir+'_brain_mask.nii.gz',
                shell=True)
            brain_mask_nib = nib.load(subj_mask_dir)
            brain_mask = brain_mask_nib.get_fdata()
            img_t2_brain = img_t2_orig * brain_mask
        else:
            img_t2_brain = img_t2_orig
        img_t2_brain_ants = ants.from_numpy(
            img_t2_brain, *args_t2_orig_ants)

        # load t1 image if exists
        if t1_exists:
            print('Load T1 image ...', end=' ')
            # copy T1 image to output directory
            subprocess.run(
                'cp '+subj_t1_dir+' '+subj_out_dir+'_T1w.nii.gz',
                shell=True)
            img_t1_orig_ants = ants.image_read(subj_t1_dir)
            img_t1_orig = img_t1_orig_ants.numpy()
            print('Done.')

            # compute T1-to-T2 ratio
            print('Compute T1/T2 ratio ...', end=' ')
            img_t1_t2_ratio = (
                img_t1_orig / (img_t2_orig+1e-12)).clip(0,100)
            save_numpy_to_nifti(
                img_t1_t2_ratio, affine_t2_orig,
                subj_out_dir+'_T1wDividedByT2w.nii.gz')
            print('Done.')

            
        # ------ Affine Registration ------
        print('--------------------')
        print('Affine registration starts ...')
        t_align_start = time.time()

        # ants affine registration
        img_t2_align_ants, affine_t2_align, trans_rigid,\
        trans_affine = registration(
            img_move_ants=img_t2_brain_ants,
            img_fix_ants=img_t2_atlas_ants,
            affine_fix=affine_t2_atlas,
            out_prefix=subj_out_dir,
            max_iter=max_regist_iter,
            min_dice=min_regist_dice,
            verbose=verbose)

        # args for converting numpy array to ants image
        args_t2_align_ants = (
            img_t2_align_ants.origin,
            img_t2_align_ants.spacing,
            img_t2_align_ants.direction)
        img_t2_align = img_t2_align_ants.numpy()

        t_align_end = time.time()
        t_align = t_align_end - t_align_start
        print('Affine registration ends. Runtime: {} sec.'.format(
            np.round(t_align, 4)))


        # ------ Cortical Ribbon Seg ------
        print('--------------------')
        print('Cortical ribbon seg starts ...')
        t_ribbon_start = time.time()

        # input volume for nn model
        vol_t2_align = torch.Tensor(img_t2_align[None,None]).to(device)
        vol_t2_align = (vol_t2_align / vol_t2_align.max()).float()
        vol_in = vol_t2_align.clone()

        # predict cortical ribbon
        with torch.no_grad():
            ribbon_pred = torch.sigmoid(seg_ribbon(vol_in))
        ribbon_align = ribbon_pred[0,0].cpu().numpy()
        ribbon_align_ants = ants.from_numpy(
            ribbon_align, *args_t2_align_ants)

        # transform back to original space
        ribbon_orig_ants = ants.apply_transforms(
            fixed=img_t2_atlas_ants,
            moving=ribbon_align_ants,
            transformlist=trans_affine['invtransforms'],
            whichtoinvert=[True],
            interpolator='linear')
        ribbon_orig_ants = ants.apply_transforms(
            fixed=img_t2_orig_ants,
            moving=ribbon_orig_ants,
            transformlist=trans_rigid['invtransforms'],
            whichtoinvert=[True],
            interpolator='linear')

        # threshold to binary mask
        ribbon_orig = ribbon_orig_ants.numpy()
        ribbon_orig = (ribbon_orig > 0.5).astype(np.float32)

        # save ribbon file
        save_numpy_to_nifti(
            ribbon_orig, affine_t2_orig, subj_out_dir+'_ribbon.nii.gz')

        t_ribbon_end = time.time()
        t_ribbon = t_ribbon_end - t_ribbon_start
        print('Cortical ribbon seg ends .... Runtime: {} sec.'.format(
            np.round(t_ribbon, 4)))


        for surf_hemi in ['left', 'right']:    
            # ------ Surface Reconstruction ------
            print('--------------------')
            print('Surface reconstruction ({}) starts ...'.format(surf_hemi))
            t_surf_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                surf_recon_wm = surf_recon_left_wm
                surf_recon_pial = surf_recon_left_pial
                # clip the left hemisphere
                vol_in = vol_t2_align[:,:,64:]
                vert_in = vert_left_in
                face_in = face_left_in
            elif surf_hemi == 'right':
                surf_recon_wm = surf_recon_right_wm
                surf_recon_pial = surf_recon_right_pial
                # clip the right hemisphere
                vol_in = vol_t2_align[:,:,:112]
                vert_in = vert_right_in
                face_in = face_right_in

            # wm and pial surfaces reconstruction
            with torch.no_grad():
                vert_wm = surf_recon_wm(vert_in, vol_in, n_steps=7)
                vert_wm = cot_laplacian_smooth(vert_wm, face_in, n_iters=1)
                vert_pial = surf_recon_pial(vert_wm, vol_in, n_steps=7)
                vert_pial = laplacian_smooth(vert_pial, face_in, n_iters=1)

            # torch.Tensor -> numpy.array
            vert_wm_align = vert_wm[0].cpu().numpy()
            vert_pial_align = vert_pial[0].cpu().numpy()
            face_align = face_in[0].cpu().numpy()

            # transform vertices to original space
            if surf_hemi == 'left':
                # pad the left hemisphere to full brain
                vert_wm_orig = vert_wm_align + [64,0,0]
                vert_pial_orig = vert_pial_align + [64,0,0]
            elif surf_hemi == 'right':
                vert_wm_orig = vert_wm_align.copy()
                vert_pial_orig = vert_pial_align.copy()
            vert_wm_orig = apply_affine_mat(
                vert_wm_orig, affine_t2_align)
            vert_pial_orig = apply_affine_mat(
                vert_pial_orig, affine_t2_align)
            face_orig = face_align[:,[2,1,0]]
            # midthickness surface
            vert_mid_orig = (vert_wm_orig + vert_pial_orig)/2

            # save as .surf.gii
            save_gifti_surface(
                vert_wm_orig, face_orig,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_wm.surf.gii',
                surf_hemi=surf_hemi, surf_type='wm')
            save_gifti_surface(
                vert_pial_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_pial.surf.gii',
                surf_hemi=surf_hemi, surf_type='pial')
            save_gifti_surface(
                vert_mid_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii',
                surf_hemi=surf_hemi, surf_type='midthickness')

            # send to gpu for the following processing
            vert_wm = torch.Tensor(vert_wm_orig).unsqueeze(0).to(device)
            vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
            vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
            face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

            t_surf_end = time.time()
            t_surf = t_surf_end - t_surf_start
            print('Surface reconstruction ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_surf, 4)))


            # ------ Surface Inflation ------
            print('--------------------')
            print('Surface inflation ({}) starts ...'.format(surf_hemi))
            t_inflate_start = time.time()

            # create inflated and very_inflated surfaces
            # if device is cpu, use wb_command for inflation (faster)
            if device == 'cpu':
                vert_inflated_orig, vert_vinflated_orig = \
                wb_generate_inflated_surfaces(
                    subj_out_dir, surf_hemi, iter_scale=3.0)
            else:  # cuda acceleration
                vert_inflated, vert_vinflated = generate_inflated_surfaces(
                    vert_mid, face, iter_scale=3.0)
                vert_inflated_orig = vert_inflated[0].cpu().numpy()
                vert_vinflated_orig = vert_vinflated[0].cpu().numpy()

            # save as .surf.gii
            save_gifti_surface(
                vert_inflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='inflated')
            save_gifti_surface(
                vert_vinflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='vinflated')

            t_inflate_end = time.time()
            t_inflate = t_inflate_end - t_inflate_start
            print('Surface inflation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_inflate, 4)))


            # ------ Spherical Mapping ------
            print('--------------------')
            print('Spherical mapping ({}) starts ...'.format(surf_hemi))
            t_sphere_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                sphere_proj = sphere_proj_left
                vert_sphere_in = vert_sphere_left_in
                bc_coord = bc_coord_left
                face_id = face_left_id
            elif surf_hemi == 'right':
                sphere_proj = sphere_proj_right
                vert_sphere_in = vert_sphere_right_in
                bc_coord = bc_coord_right
                face_id = face_right_id

            # interpolate to 160k template
            vert_wm_160k = (vert_wm_orig[face_id] * bc_coord[...,None]).sum(-2)
            vert_wm_160k = torch.Tensor(vert_wm_160k[None]).to(device)
            feat_160k = torch.cat([vert_sphere_160k, vert_wm_160k], dim=-1)

            with torch.no_grad():
                vert_sphere = sphere_proj(
                    feat_160k, vert_sphere_in, n_steps=7)
                
            # compute metric distortion
            edge = torch.cat([
                face[0,:,[0,1]],
                face[0,:,[1,2]],
                face[0,:,[2,0]]], dim=0).T
            edge_distort = 100. * edge_distortion(
                vert_sphere, vert_wm, edge).item()
            area_distort = 100. * area_distortion(
                vert_sphere, vert_wm, face).item()
            print('Edge distortion: {}%'.format(np.round(edge_distort, 2)))
            print('Area distortion: {}%'.format(np.round(area_distort, 2)))
            
            # save as .surf.gii
            vert_sphere = vert_sphere[0].cpu().numpy()
            save_gifti_surface(
                vert_sphere, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sphere.surf.gii',
                surf_hemi=surf_hemi, surf_type='sphere')

            t_sphere_end = time.time()
            t_sphere = t_sphere_end - t_sphere_start
            print('Spherical mapping ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_sphere, 4)))


            # ------ Cortical Feature Estimation ------
            print('--------------------')
            print('Feature estimation ({}) starts ...'.format(surf_hemi))
            t_feature_start = time.time()

            print('Estimate cortical thickness ...', end=' ')
            thickness = cortical_thickness(vert_wm, vert_pial)
            thickness = metric_dilation(
                torch.Tensor(thickness[None,:,None]).to(device),
                face, n_iters=10)
            save_gifti_metric(
                metric=thickness,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii',
                surf_hemi=surf_hemi, metric_type='thickness')
            print('Done.')

            print('Estimate curvature ...', end=' ')
            curv = curvature(vert_wm, face, smooth_iters=5)
            save_gifti_metric(
                metric=curv, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_curv.shape.gii',
                surf_hemi=surf_hemi, metric_type='curv')
            print('Done.')


            print('Estimate sulcal depth ...', end=' ')
            sulc = sulcal_depth(vert_wm, face, verbose=False)
            save_gifti_metric(
                metric=sulc,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sulc.shape.gii',
                surf_hemi=surf_hemi, metric_type='sulc')
            print('Done.')

            # estimate myelin map based on
            # t1-to-t2 ratio, midthickness surface, 
            # cortical thickness and cortical ribbon

            if t1_exists:
                print('Estimate myelin map ...', end=' ')
                myelin, smoothed_myelin = myelin_map(
                    subj_dir=subj_out_dir, surf_hemi=surf_hemi)
                myelin = metric_dilation(
                    torch.Tensor(myelin[None,:,None]).to(device),
                    face, n_iters=10)
                smoothed_myelin = metric_dilation(
                    torch.Tensor(smoothed_myelin[None,:,None]).to(device),
                    face, n_iters=10)
                # save myelin map
                save_gifti_metric(
                    metric=myelin, 
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii',
                    surf_hemi=surf_hemi, metric_type='myelinmap')
                save_gifti_metric(
                    metric=smoothed_myelin, 
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+\
                             '_smoothed_myelinmap.shape.gii',
                    surf_hemi=surf_hemi,
                    metric_type='smoothed_myelinmap')
                print('Done.')

            t_feature_end = time.time()
            t_feature = t_feature_end - t_feature_start
            print('Feature estimation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_feature, 4)))

        print('--------------------')
        # clean temp data
        os.remove(subj_out_dir+'_rigid_0GenericAffine.mat')
        os.remove(subj_out_dir+'_affine_0GenericAffine.mat')
        os.remove(subj_out_dir+'_ribbon.nii.gz')
        # create .spec file for visualization
        create_wb_spec(subj_out_dir)
        t_end = time.time()
        print('Finished. Total runtime: {} sec.'.format(
            np.round(t_end-t_start, 4)))
        print('====================')

        