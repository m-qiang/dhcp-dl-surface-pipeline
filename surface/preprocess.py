import sys
import os
os.chdir('..')
sys.path.append(os.getcwd())

import shutil
import nibabel as nib
import glob
import argparse
import ants
import numpy as np
from tqdm import tqdm
from utils.register import registration
from utils.io import save_numpy_to_nifti, save_gifti_surface
import pymeshlab



def split_data(orig_dir, save_dir, seed=12345):
    """split original dataset"""

    subj_list = sorted(os.listdir(orig_dir))
    print("Number of all data:", len(subj_list))

    # ------ randomly split train/valid/test data ------ 
    np.random.seed(seed)
    subj_permute = np.random.permutation(len(subj_list))
    n_train = int(len(subj_list) * 0.6)
    n_valid = int(len(subj_list) * 0.1)
    n_test = len(subj_list) - n_train - n_valid
    print("Number of training data:", n_train)
    print("Number of validation data:", n_valid)
    print("Number of testing data:", n_test)

    train_list = subj_permute[:n_train]
    valid_list = subj_permute[n_train:n_train+n_valid]
    test_list = subj_permute[n_train+n_valid:]
    data_list = [train_list, valid_list, test_list]
    data_split = ['train', 'valid', 'test']

    for n in range(3):
        for i in data_list[n]:
            subj_id = subj_list[i]
            subj_dir = save_dir+data_split[n]+'/'+subj_id
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)
                
                
                
def remesh(vert, face, n_target=160000):

    # compute target edge length
    edge = np.concatenate([
        face[:,[0,1]], face[:,[1,2]], face[:,[2,0]]], axis=0).T
    edge_len = np.sqrt(((vert[edge[0]] - vert[edge[1]])**2).sum(-1)).mean()
    target_len = edge_len * np.sqrt(vert.shape[0] / n_target)

    # remesh the surface
    m = pymeshlab.Mesh(vert, face)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5, adaptive=False, checksurfdist=False,
        targetlen=pymeshlab.AbsoluteValue(target_len))

    vert_remesh = ms[0].vertex_matrix()
    face_remesh = ms[0].face_matrix()
    
    print("num of vertices before/after: {} / {}".format(
        vert.shape[0], vert_remesh.shape[0]))
    return vert_remesh, face_remesh
    
    

def process_data(orig_dir, save_dir, t2_suffix, mask_suffix=None):
    data_split = ['train', 'valid', 'test']

    img_fix_ants = ants.image_read(
        './template/dhcp_week-40_template_T2w.nii.gz')
    affine_fix = nib.load(
        './template/dhcp_week-40_template_T2w.nii.gz').affine
    img_fix = img_fix_ants
    
    # minimum dice score for registration
    min_dice=0.9
    
    for n in range(3):
        subj_list = sorted(os.listdir(save_dir+data_split[n]))
        for subj_id in tqdm(subj_list):
            subj_orig_dir = orig_dir+subj_id+'/'+subj_id
            subj_save_dir = save_dir+data_split[n]+'/'+subj_id+'/'+subj_id
            print("process subject:", subj_id)

            # ------ affine align t2w image ------
            # t2 image after bias field correction
            img_restore_nib = nib.load(subj_orig_dir+t2_suffix)
            img_restore = img_restore_nib.get_fdata()

            # load brain mask
            if mask_suffix:
                brain_mask_nib = nib.load(subj_orig_dir+mask_suffix)
                brain_mask = brain_mask_nib.get_fdata()
                img_restore_brain = img_restore * brain_mask
            else:
                img_restore_brain = img_restore
                
            # use restored brain-extracted image
            img_move_ants = ants.image_read(subj_orig_dir+t2_suffix)
            img_move_ants = ants.from_numpy(
                img_restore_brain,
                origin=img_move_ants.origin,
                spacing=img_move_ants.spacing,
                direction=img_move_ants.direction)

            # affine registration
            img_align_ants, affine_align, _, _, align_dice = registration(
                img_move_ants, img_fix_ants, affine_fix,
                out_prefix=subj_save_dir, min_dice=min_dice)
            if align_dice >= min_dice:
                logger.info('Dice after registration: {}'.format(align_dice))
            else:
                logger.info('Error! Affine registration failed!')
                logger.info('Expected Dice>{} after registraion, got Dice={}.'.format(
                    min_dice, align_dice))
            os.remove(subj_save_dir+'_rigid_0GenericAffine.mat')
            os.remove(subj_save_dir+'_affine_0GenericAffine.mat')

            img_align = img_align_ants.numpy()
            # save t2w image 
            save_numpy_to_nifti(img_align.astype(np.float32),
                                affine_align,
                                subj_save_dir+'_T2w_affine.nii.gz')
            
            # ------ surface remesh ------
            # copy surfaces
            for surf_hemi in ['left', 'right']:
                for surf_type in ['wm', 'pial']:
                    surf_orig_dir = subj_orig_dir+'_hemi-'+surf_hemi+'_'+surf_type+'.surf.gii'
                    surf_save_dir = subj_save_dir+'_hemi-'+surf_hemi+'_'+surf_type+'.surf.gii'
                    shutil.copy(surf_orig_dir, surf_save_dir)
                    surf = nib.load(surf_save_dir) 
                    vert = surf.agg_data('pointset')
                    face = surf.agg_data('triangle')
                    # remesh the surface meshes to 150-160k vertices for training
                    vert_150k, face_150k = remesh(vert, face, n_target=160000)
                    save_gifti_surface(
                        vert_150k, face_150k,
                        save_dir=surf_save_dir[:-9]+'_150k.surf.gii',
                        surf_hemi=surf_hemi, surf_type=surf_type)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Data Proprocessing")

    parser.add_argument('--orig_dir', default='../YOUR_DATA/', type=str, help="directory of original dataset")
    parser.add_argument('--save_dir', default='./surface/data/', type=str, help="directory for saving processed data")
    parser.add_argument('--T2', default='_T2w.nii.gz', type=str, help='Suffix of T2 image file.')
    parser.add_argument('--mask', default=None, type=str, help='Suffix of brain mask file.')
    
    args = parser.parse_args()
    orig_dir = args.orig_dir
    save_dir = args.save_dir
    t2_suffix = args.T2
    mask_suffix = args.mask
    
    split_data(orig_dir, save_dir)
    process_data(orig_dir, save_dir, t2_suffix, mask_suffix)