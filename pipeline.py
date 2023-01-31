# -*- coding: utf-8 -*-
"""
Created on Oct 20 2022
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering, University of Washington, Seattle, USA
@Refs:
    1. SynthStrip:
        Hoopes, A., Mora, J. S., Dalca, A. V., Fischl, B., &amp; Hoffmann, M. (2022). Synthstrip: 
        Skull-stripping for any brain image. NeuroImage, 260, 119474. https://doi.org/10.1016/j.neuroimage.2022.119474 
    2. SimpleITK:
        R. Beare, B. C. Lowekamp, Z. Yaniv, “Image Segmentation, Registration and Characterization 
        in R with SimpleITK”, J Stat Softw, 86(8), doi: 10.18637/jss.v086.i08, 2018.
"""

import os
import subprocess
import pandas as pd
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
from matplotlib import pyplot as plt
import pyautogui
from scipy.ndimage import zoom
import SimpleITK as sitk
from skimage import exposure
from tqdm import tqdm
from scipy import signal


def contrast_correction(movingNorm, staticNorm, percentile = (2, 98)):
    '''
    Exposure correction and histogram exposure matching
    Parameters
    ----------
    movingNorm : numpy array
        normalized moving image.
    staticNorm :  numpy array
        normalized static image.
    percentile : tuple, optional
        range to keep(outliar removal). The default is (2, 98).
    Returns
    -------
    matched : numpy array
        corrected moving image.
    static_rescale : TYPE
        corrected static image.
    '''
    p1, p99 = np.percentile(staticNorm, percentile)
    static_rescale = exposure.rescale_intensity(staticNorm, in_range=(p1, p99))
    
    p1, p99 = np.percentile(movingNorm, percentile)
    moving_rescale = exposure.rescale_intensity(movingNorm, in_range=(p1, p99))
    
    matched = exposure.match_histograms(moving_rescale, static_rescale)
    
    return matched, static_rescale


def denoise(A,alpha=1E-3):
    '''
    De-noise the data

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1E-3.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    '''
    mask = np.where(A > alpha, 1, 0)
    A *= mask
    return A


def norm(A):
    '''
    Normalize/rescale numpy array to [0,1]
    Parameters
    ----------
    A : numpy array
    Returns
    -------
    numpy array
        normalized array.
    '''
    return (A - np.min(A))/(np.max(A) - np.min(A))


def smooth(data, kernel_size = 7):
    '''
    Applies a gaussian smoothning filter

    Parameters
    ----------
    data : numpy array
        3d input array.
    kernel_size : int, optional
        size of the kernel/window size (odd values only). The default is 7.

    Returns
    -------
    filtered : numpy array
        smoothned 3d image.

    '''
    sigma = 1.0     # width of kernel
    x = np.arange(-int(kernel_size/2),int(kernel_size/2)+int(kernel_size%2),1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-int(kernel_size/2),int(kernel_size/2)+int(kernel_size%2),1)
    z = np.arange(-int(kernel_size/2),int(kernel_size/2)+int(kernel_size%2),1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    filtered = signal.convolve(data, kernel, mode="same")
    return filtered
    
    
def PET_average(in_folder_path,out_folder_path = None,file_name = 'PET_averaged.nii', add_only = False):
    '''
    Average multiple PET snapshots to get an averaged snapshot over the whole timeperiod.
    Parameters
    ----------
    in_folder_path : STRING
        PATH of the input folder.
    out_folder_path : STRING
        PATH of the output folder.
    file_name : STRING, optional
        Name of the combined image. The default is 'PET_averaged.nii'.
    Returns
    -------
    data : numpy array
        averaged pet scans.
    meta : list
        meta data for all pet files.
    '''
    A = os.path.join(in_folder_path,os.listdir(in_folder_path)[0])
    B = os.path.join(A, os.listdir(A)[0])
    scans = os.listdir(B)
    data = []
    meta = []
    
    for scan in scans:
        filename = os.path.join(data_path, os.path.join(B, scan))
        img = nib.load(filename)
        dt = img.get_fdata()
        data.append(dt)
        meta.append(img.header)
        
    data = np.squeeze(np.array(data))
    
    
    #[18F]-AV-1451 PET acquisition (ET: 80-100 min) post injection.
    #ADNI3 acquires a 30 min dynamic scan consisting of six 5-minute frames. Acquisition starts promptly at 75 minutes post injection. 
    if len(scans) > 1: # else 1 5m range somewhere within 75-105min
        #co-register data
        if len(scans) <= 4:#assuming range within 75-105
            data = np.array([data[0]] + [affine_register(data[0], p) for p in data[1:]])
        elif len(scan) < 6:#assuming 80-100 or 85-105
            data = np.array([data[1]] + [affine_register(data[1], p) for p in data[2:]])
        else:#range 80-100min
            data = np.array([data[1]] + [affine_register(data[1], p) for p in data[2:-1]])
         
        if add_only == False:
            data = np.mean(data,axis=0)
        else:
            data = np.sum(data,axis=0)
          
    if out_folder_path is not None:
        img = nib.Nifti1Image(data, affine = np.eye(4))
        nib.save(img, os.path.join(out_folder_path,file_name))    
    else:   
        return data, meta


def skullstrip(sudo_password, folder_path, output_path, input_path, mask = False):    
    '''
    Wraper function to skull strip various imaging modalities using SynthStrip's command line docker command.
        Hoopes, A., Mora, J. S., Dalca, A. V., Fischl, B., &amp; Hoffmann, M. (2022). Synthstrip: Skull-stripping for any brain image. 
        NeuroImage, 260, 119474. https://doi.org/10.1016/j.neuroimage.2022.119474 
    Parameters
    ----------
    sudo_password : string
        Password for sudo privilage if any.
    folder_path : string
        path of the root folder in the llocal system.
    outputh_path : string
        path of the output .nii skull stripped image.
    input_path : string, optional
        input path of the .nii image. The default is 'temp.nii'.
    Returns
    -------
    output : string
        output of the command in terminal.
    '''
    #TO DO: option to run on gpu...
    if mask:
        command = 'sudo docker run -v %s:/home/temp/ freesurfer/synthstrip -i %s -m %s'%(
            folder_path,'/home/temp/'+input_path+'.nii','/home/temp/'+output_path+'.nii')
    else:
        command = 'sudo docker run -v %s:/home/temp/ freesurfer/synthstrip -i %s -o %s'%(
            folder_path,'/home/temp/'+input_path+'.nii','/home/temp/'+output_path+'.nii')
    
    command = command.split()
    
    cmd1 = subprocess.Popen(['echo',sudo_password], stdout=subprocess.PIPE)
    cmd2 = subprocess.Popen(['sudo','-S'] + command, stdin=cmd1.stdout, stdout=subprocess.PIPE)
    
    output = cmd2.stdout.read().decode() 
    
    return output


def cp2dir(input_folder, output_folder, output_name,sudo_password):
    '''
    Copy a file from one dir to another with a new name.
    Parameters
    ----------
    input_folder : STRING
        path of input folder.
    output_folder : STRING
        path of output folder.
    output_name : STRING
        name of output file.
    sudo_password : STRING
        sudo password.
    Returns
    -------
    output : STRING
        console output.
    '''
    output = []
    A = input_folder
    B = os.path.join(A, os.listdir(A)[-1])
    C = os.path.join(B, os.listdir(B)[-1])
    
    for i,file in enumerate(os.listdir(C)):
        D = os.path.join(C ,file)
        
        E = os.path.join(output_folder, output_name + '.nii') if len(
            os.listdir(C)) == 1 else os.path.join(output_folder, output_name + str(i+1) + '.nii')
        #print('###Copying ' + D + ' to ' + E)
        
        command = 'sudo cp %s %s'%(D,E)
        
        command = command.split()
        
        cmd1 = subprocess.Popen(['echo',sudo_password], stdout=subprocess.PIPE)
        cmd2 = subprocess.Popen(['sudo','-S'] + command, stdin=cmd1.stdout, stdout=subprocess.PIPE)
        
        output.append(cmd2.stdout.read().decode())
        #print(output[-1])
    
    return output


def quick_mask(A, password, temp_folder = '/home/agam/Documents/temp', get_mask = False):
    '''
    Generate mask for input volume
    Parameters
    ----------
    A : numpy array
        brain volume to be masked.
    password : string
        sudo password.
    temp_folder : string, optional
        path to a temp folder in system. The default is '/home/agam/Documents/temp'.
    get_mask : bool, optional
        pick if output is just the mask or mask applied to the input volume. The default is False.
    Returns
    -------
    mask or mask applied to input volume
    '''
    img = nib.Nifti1Image(A, affine = np.eye(4))
    
    nib.save(img, os.path.join(temp_folder,'temp.nii'))
    
    skullstrip(password, temp_folder,'temp_mask','temp',True)
    
    if get_mask:
        return np.squeeze(nib.load(os.path.join(temp_folder,'temp_mask.nii')).get_fdata())
    
    else:
        return A * np.squeeze(nib.load(os.path.join(temp_folder,'temp_mask.nii')).get_fdata())


def zero_crop(A, mask, threshold = 0.01):
    '''
    Crop out zero regions of a volume
    Parameters
    ----------
    A : numpy array
        input volume.
    mask : numpy array
        mask associating input volume, region to be kept.
    threshold : float, optional
        if mask is not binary, use threshold to generate on-the-fly binary mask. The default is 0.01.
    Returns
    -------
    scan : numpy array
        cropped volume of interest.
    '''
    scan = np.where(mask > threshold, 1, 0)
    
    x_ = np.squeeze(np.where(np.sum(np.sum(scan,axis=2),axis=1) > 0.))
    x_start = int(x_[0])
    x_end = int(x_[-1])

    y_ = np.squeeze(np.where(np.sum(np.sum(scan,axis=2),axis=0) > 0.))
    y_start = int(y_[0])
    y_end = int(y_[-1])

    z_ = np.squeeze(np.where(np.sum(np.sum(scan,axis=1),axis=0) > 0.))
    z_start = int(z_[0])
    z_end = int(z_[-1])

    scan = A[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]
    
    return scan


def center_crop3d(A,target_shape):
    '''
    Lazy cropping, center crop to some target shape
    Parameters
    ----------
    A : numpy array
        input volume.
    target_shape : tuple
        desired shape of the output volume.
    Returns
    -------
    B : numpy array
        cropped output volume.
    '''
    input_shape = A.shape
    starts = [input_shape[i]//2 - (target_shape[i]//2) for i in range(len(target_shape))]
    
    B = A[starts[0]:starts[0]+target_shape[0],starts[1]:starts[1]+target_shape[1],
          starts[2]:starts[2]+target_shape[2]]
  
    return B


def roi_crop(scans = [], password = ''):
    '''
    Pipeline for cropping RoIs for multimodal inputs
    Parameters
    ----------
    scans : list of numpy arrays, optional
        list of multimodal voumes. The default is [].
    password : string, optional
        sudo password. The default is ''.
    Returns
    -------
    scans : list of numpy arrays
        RoI cropped output volumes.
    '''
    shapes = np.asarray([scan.shape for scan in scans])
    #print('input shapes:')
    #print(shapes)
    
    target_shape = np.min(shapes,axis=0)
    #print('target_shape:', target_shape)
    
    scans = [center_crop3d(scan, target_shape) for scan in scans] 
    
    masked_scans = [quick_mask(scan, password) for scan in scans]
    
    scans = masked_scans#[masked_scans[0], masked_scans[1], scans[2]]    
    scans = [zero_crop(scans[i], masked_scans[i]) for i in range(len(scans))]
    #scans = [zero_crop(scan, scan) for scan in masked_scans]
    
    #plot_scans(scans)
    
    return scans


def plot_scans(scans = []):
    '''
    plots list of 3d volumes
    Parameters
    ----------
    scans : list of numpy arrays, optional
        list of input volumes to be plotted. The default is [].
    Returns
    -------
    None.
    '''
    c = len(scans[0].shape)
    r = len(scans)
    i = 0
    
    fig = plt.figure(figsize = (15,15), dpi = 180)
    
    for scan in scans:
        fig.add_subplot(r,c,i+1)
        plt.imshow(scan[int(scan.shape[0]/2)],cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=.05)
        plt.axis('off') 
        
        fig.add_subplot(r,c,i+2)
        plt.imshow(scan[:,int(scan.shape[1]/2)],cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=.05)
        plt.axis('off')
        
        fig.add_subplot(r,c,i+3)
        plt.imshow(scan[:,:,int(scan.shape[2]/2)],cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=.05)
        plt.axis('off')
        
        i += 3

    plt.show()


def resampleA2B(A, A_vox_dim, B_vox_dim = (1., 1., 1.)):
    '''
    resample volume A using voxel lengths of A to that of a refrance volume B assuming 0 slice spacing
    Parameters
    ----------
    A : numpy array
        input volume.
    A_vox_dim : tuple
        voxel lengths of A.
    B_vox_dim : tuple, optional
        voxel lengths of a refrance volume. The default is (1., 1., 1.).
    Returns
    -------
    resampled_A : numpy array
        resampled volume A w.r.t. B.
    '''
    #print('input shape:', A.shape)   
    vox_dim = (A_vox_dim[0]/B_vox_dim[0], A_vox_dim[1]/B_vox_dim[1], A_vox_dim[2]/B_vox_dim[2])    
    #print('transformation factors:', vox_dim)  
    resampled_A = np.squeeze(zoom(A, vox_dim, mode = 'nearest'))   
    #print('output shape:', resampled_A.shape)  
    
    return resampled_A


def reshapeA2B(A, target_shape = (176,176,176)):
    '''
    Reshape input volume to a target shape
    Parameters
    ----------
    A : numpy array
        input volume.
    target_shape : tuple, optional
        target shape. The default is (176,176,176).
    Returns
    -------
    resampled_A : numpy array
        resampled volume A of desired target shape.
    '''
    #print('input shape:', A.shape) 
    in_shape = A.shape
    vox_dim = (target_shape[0]/in_shape[0], target_shape[1]/in_shape[1], 
               target_shape[2]/in_shape[2])    
    #print('transformation factors:', vox_dim)  
    resampled_A = np.squeeze(zoom(A, vox_dim, mode = 'nearest'))   
    #print('output shape:', resampled_A.shape)  
    
    return resampled_A
    
 
def rigid_register(fixed, moving, prt = False, bins = 50, sp = 0.01):
    '''
    Simple Rigid registration using SimpleITK
    Parameters
    ----------
    fixed : numpy array
        fixed volume.
    moving : numpy array
        moving volume.
    Returns
    -------
    moving_resampled : numpy array
        registered moving image w.r.t. fixed volume.
    '''
    fixed_image = sitk.GetImageFromArray(fixed)
    fixed_image.SetOrigin((0, 0, 0))   
    
    moving_image = sitk.GetImageFromArray(moving)
    moving_image.SetOrigin((0, 0, 0))
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sp)
    
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()#Scale voxel translation(mm) and rotation(theta) correctly
    
    # Setup for the multi-resolution framework. # Only turn this on if simple registration doesnt work.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Always check the reason optimization terminated.
    if prt:
        print("Final metric value: {0}".format(registration_method.GetMetricValue()))
        print("Optimizer's stopping condition, {0}".format(
            registration_method.GetOptimizerStopConditionDescription()))
    
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, 
                                     sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())                         
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    
    return moving_resampled
 
    
def affine_register(fixed, moving, prt = False, bins = 50, sp = 0.01):
    '''
    Simple Affine registration using SimpleITK
    Parameters
    ----------
    fixed : numpy array
        fixed volume.
    moving : numpy array
        moving volume.
    Returns
    -------
    moving_resampled : numpy array
        registered moving image w.r.t. fixed volume.
    '''
    fixed_image = sitk.GetImageFromArray(fixed)
    fixed_image.SetOrigin((0, 0, 0))
    
    moving_image = sitk.GetImageFromArray(moving)
    moving_image.SetOrigin((0, 0, 0))
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                  moving_image,
                                                  sitk.AffineTransform(fixed_image.GetDimension()),
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sp)
    
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()#Scale voxel translation(mm) and rotation(theta) correctly
    
    # Setup for the multi-resolution framework. # Only turn this on if simple registration doesnt work.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Always check the reason optimization terminated.
    if prt:
        print("Final metric value: {0}".format(registration_method.GetMetricValue()))
        print("Optimizer's stopping condition, {0}".format(
            registration_method.GetOptimizerStopConditionDescription()))
    
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, 
                                     sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())                         
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    
    return moving_resampled
    

def preprocess_pipeline(folder_path,rpath_T1,rpath_T2,rpath_PET,rpath_out_folder,idx,password,temp_folder):  
    '''
    Complete preprocessing pipeline for ADNI3 dataset for multi-modal(T1,T2_FLAIR) MRI and PET registration.
    Parameters
    ----------
    folder_path : string
        path of root folder.
    rpath_T1 : string
        relative path of T1 folder.
    rpath_T2 : string
        relative path of T2 folder.
    rpath_PET : string
        relative path of PET folder.
    rpath_out_folder : string
        relative path of output folder.
    idx : int
        desired index of output data.
    password : string
        sudo password.
    temp_folder : string
        full path of temp folder
    Returns
    -------
    None.
    '''    
    try:
        nib.load(os.path.join(folder_path,os.path.join(rpath_out_folder,'T1_%d.nii'%(idx))))
        nib.load(os.path.join(folder_path,os.path.join(rpath_out_folder,'T2_%d.nii'%(idx))))
        nib.load(os.path.join(folder_path,os.path.join(rpath_out_folder,'PET_%d.nii'%(idx))))
        
    except:
        #Loading Data
        #print('loading PET')
        try:
            PET, PET_meta = PET_average(os.path.join(folder_path,rpath_PET), add_only = False)
        except:
            PET, PET_meta = PET_average(os.path.join(folder_path,rpath_PET + '_Tau'), add_only = False)
        PET, PET_meta = np.squeeze(PET), PET_meta[-1]
        
        #print('loading T2')
        A = os.path.join(folder_path,rpath_T2)
        B = os.path.join(A, os.listdir(A)[0])
        C = os.path.join(B, os.listdir(B)[0])
        D = os.path.join(C, os.listdir(C)[0])
        #print(D)
        img = nib.load(D)
        T2, T2_meta = np.squeeze(img.get_fdata()), img.header
        
        #print('loading T1')
        A = os.path.join(folder_path,rpath_T1)
        B = os.path.join(A, os.listdir(A)[0])
        C = os.path.join(B, os.listdir(B)[0])
        D = os.path.join(C, os.listdir(C)[0])
        #print(D)
        img = nib.load(D)
        T1, T1_meta = np.squeeze(img.get_fdata()), img.header
    
        #Resample T1, T2, PET to approx (1mm x 1mm x 1mm) voxel grid
        PET = resampleA2B(PET,PET_meta.get_zooms()[:-1]) 
        T1 = resampleA2B(T1,T1_meta.get_zooms()[:-1])
        T2 = resampleA2B(T2,T2_meta.get_zooms()[:-1]) 
        
        #Smoothn PET scan
        PET = smooth(PET,7)
        
        #Normalize
        PET, T1, T2 = norm(PET), norm(T1), norm(T2)
        
        #Simple RoI cropping
        cropped = roi_crop([T1,T2,PET], password)
        T1,T2,PET = cropped[0],cropped[1],cropped[2]
        #print(T1.shape, T2.shape, PET.shape)
        
        #Histogram Correction
        T2, T1_ = contrast_correction(T2, T1)
        PET,_ = contrast_correction(PET, T1)
        T1 = T1_
        
        #Ridgid Registration ###TO DO: RUN MULTIPLE TRIALS WITH DIFFERENT PARAMETERS
        T2 = rigid_register(T1, T2)
        PET = rigid_register(T1, PET)
        
        #Affine Registration ###TO DO: RUN MULTIPLE TRIALS WITH DIFFERENT PARAMETERS
        T2 = affine_register(T1, T2)
        PET = affine_register(T1, PET)
        
        #SkullStrip and Masking
        mask = quick_mask(A = T1, password = password, get_mask=True)
        T1 *= mask
        T2 *= mask
        PET *= mask
        
        #Reshaping to standard cube for training models
        target_shape = (176,176,176)
        T1,T2,PET = reshapeA2B(T1, target_shape), reshapeA2B(T2, target_shape), reshapeA2B(PET, target_shape)
        
        #Remove noise artifacts from preprocessing
        T1,T2,PET = denoise(T1),denoise(T2),denoise(PET)
        
        #Save preprocessing outputs as NIFTI
        #print('saving...')
        imgT1 = nib.Nifti1Image(T1, affine = np.eye(4))
        imgT2 = nib.Nifti1Image(T2, affine = np.eye(4))
        imgPET = nib.Nifti1Image(PET, affine = np.eye(4))
        
        nib.save(imgT1, os.path.join(folder_path,os.path.join(rpath_out_folder,'T1_%d.nii'%(idx))))
        nib.save(imgT2, os.path.join(folder_path,os.path.join(rpath_out_folder,'T2_%d.nii'%(idx))))
        nib.save(imgPET, os.path.join(folder_path,os.path.join(rpath_out_folder,'PET_%d.nii'%(idx))))
        
        #Sanity check
        plot_scans([T1, T2, PET])


def get_usable_subjects(csv):
    '''
    Generate a dictionary of usable subjects based on some criteria.
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    Returns
    -------
    usable_subjects : dictionary
        usable subjects.
    usable : int
        number of usable subjects.
    '''
    subjects = np.array(csv['Subject'])
    return subjects


def generate_address_dict(matched_csv, mri_csv, pet_csv):#{'PID':[T1_folder,T2_folder,PET_folder],...}
    '''
    Generate dictionary of usable patient ids with T1, T2-FLAIR, and PET folders as in the ADNI3 csv file
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    Returns
    -------
    data : dictonary
        dictionary of modality file names by patient ids extracted from the ADNI3 csv file.
    '''
    usable_subjects = get_usable_subjects(matched_csv)    
    data = {}
    for ids in usable_subjects:
        val = {}      
        mri = mri_csv.loc[mri_csv['Subject'] == ids].to_numpy()    
        pet = pet_csv.loc[pet_csv['Subject'] == ids].to_numpy()    
        
        for b in mri:
            if b[6] == 'MRI':
                if 'Sag' in b[7]:
                    if 'Acc' in b[7] and 'ND' not in b[7]:
                        val['T1'] = b[7].replace(' ', '_')             
                    elif 'FLAIR' in b[7]:
                        val['T2'] = b[7].replace(' ', '_')   
                        
        for b in pet:
            if b[6] == 'PET':
                val['PET'] = b[7].replace(':', '_').replace(' ', '_').replace('/','_').replace('(','_').replace(')','_').replace('_Tau','') 
        
        try:
            data[ids] = [val['T1'],val['T2'],val['PET']]
        except:
            continue
    
    return data


def get_data_address_list(csv=[], mri_path = 'ad_project/data/final_adni/mri/ADNI/', pet_path = 'ad_project/data/final_adni/pet/ADNI/'):
    '''
    Generate list of usable relative data file paths
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    file_path : string, optional
        relative folder path of target folder. The default is 'ad_project/data/initial_only/intial_only/ADNI/'.
    Returns
    -------
    data_ : list of lists
        list of relative filepaths extracted from the ADNI3 csv file.
    '''
    data= generate_address_dict(csv[0],csv[1],csv[2])
    
    keys = data.keys()    
    data_ = []
    
    for pid in keys:
        temp = []  
        for i in range(len(data[pid])):
            if i < 2:
                temp.append(mri_path + pid + '/' + data[pid][i])
            else:
                temp.append(pet_path + pid + '/' + data[pid][i])
        data_.append(temp)
   
    return data_


def main_iterator(temp = '/home/agam/Documents/temp', out = 'temp/outF', 
                 root = '/home/agam/Desktop/', matched = '/home/agam/Desktop/ad_project/data/final_adni/matched.csv', 
                 pet = '/home/agam/Desktop/ad_project/data/final_adni/pet_tau.csv', mri = '/home/agam/Desktop/ad_project/data/final_adni/mri.csv'):
    #CSV LOGIC     
    csv = []
    csv.append(pd.read_csv(matched))
    csv.append(pd.read_csv(mri))
    csv.append(pd.read_csv(pet))
    adrs = get_data_address_list(csv)
    
    password = pyautogui.password(text="[sudo] password for agam: ", title='', default='', mask='*')
      
    #preprocessing loop
    i = 0
    print('Preprocessing initiated for %d patients...'%(len(adrs)))
    
    for adr in tqdm(adrs):
        T1 = adr[0]
        T2 = adr[1]
        PET = adr[2]      
        flag = preprocess_pipeline('/home/agam/Desktop/',T1,T2,PET,out,i,password,temp)      
        i += 1
        if flag == 0:
            print('ERROR')
            break
        
    print('...done!')


if __name__ == "__main__":
    main_iterator()
