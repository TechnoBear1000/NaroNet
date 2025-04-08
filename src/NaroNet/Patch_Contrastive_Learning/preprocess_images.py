"""
This module normalizes the cohort based on the mean and standard deviation.
"""

import math
import shutil
import multiprocessing
#import multiprocessing.dummy as multiprocessing
import os
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import csv
from NaroNet.utils.parallel_process import parallel_process
import NaroNet.utils.utilz as utilz
from skimage import io, measure
import time
import tifffile
import geojson
import pandas as pd


def loadImage(path,Channels):
    '''
    path: (string) that specifies the image path
    Channels: (vector of int) channels that should be included in the experiment.    
    '''

    # Load Image in its own format.
    if path.suffix=='.tiff' or  path.suffix=='.tif':
        image = io.imread(path)    
    elif path.suffix=='.npy':
        image = np.load(path)
    else:
        raise NotImplementedError(f"Can't load unexpected file type: {image.suffix}")

    if len(image.shape)==3:
        # The 3rd dimension should be the channel dimension
        if np.argmin(image.shape)==0:
            shp = image.shape
            image = np.transpose(image, (1,2,0))
            # What is going on here, this seem wrong. Reshape does not transpose. This would 
            # essentially lead to a jumbled mess of channels and the spatial dimensions.
            #image = np.reshape(image,(image.shape[1]*image.shape[2],image.shape[0]))  
            #image = np.reshape(image,(shp[1],shp[2],shp[0]))
        elif np.argmin(image.shape)==1:
            image = np.transpose(image, (0,2,1))
            #image = np.reshape(image,(image.shape[0]*image.shape[2],image.shape[1]))
    
    elif len(image.shape)==4:
        shp = image.shape
        # This is puzzling. I would assume a 4D tensor in this case to be a batch of 
        # images, which the below would have been ok with (had it done transpose instead of reshape) but the rest 
        # of the code seem to assume the channel dimension is last, not first like here. I suspect this code was never used much.
        image = image.reshape((shp[1],shp[0],shp[2],shp[3]))  # This seems very wrong as well. How does this image actually look
    
    # The histogram calculation needs positive values. SRY. 
    # This looks very weird as well, this will calculate the minimum value along the 
    # first (height) axis _over_ channels. Surely this is wrong? I would have assumed the min would be along the channels. 
    # Also, does the histogram calculation really need positive values?
    # If any centering needs to be done, let the actual code dealing with the statistics handle that
    # image = image-image.min(tuple(range(1,len(image.shape))), keepdims=True)

    # Eliminate unwanted channels. Essentially allow you to make a subselection of all channels based on the Channels.txt file
    if len(image.shape)==3:
        return image[:,:,Channels]
    if len(image.shape)==4:
        return image[Channels,:,:,:]

def Mean_std_experiment(base_path,image_paths,Channels):
    ''' 
    Obtain mean and standard deviation from the cohort
    base_path: (string) that specifies the directory where the experiment is carried out.
    images_paths: (list of strings) that specifies the names of the files executed.
    Channels: (vector of int) channels that should be included in the experiment.    
    '''
    # Calculate the mean and standard deviation using the Welford online algorithm
    mean = np.zeros(len(Channels))
    M2 = np.zeros(len(Channels))
    n = 0
    
    for image_path in image_paths:
        image = loadImage(image_path, Channels)
        
        dimensions = image.shape[:-1]
        non_channel_ax = tuple(range(len(dimensions)))
        summed_channels = image.sum(axis=non_channel_ax)
        num_pixels = np.prod(dimensions)
        
        # Update the means
        new_n = n + num_pixels
        delta = summed_channels / num_pixels - mean
        mean += delta * num_pixels / new_n
        
        # Update M2
        M2 += np.sum((image - mean)**2, axis=non_channel_ax)
        
        n = new_n
    
    variance = M2 / n
    std_dev = np.sqrt(variance)
    return mean, std_dev
    
def Mean_std_experiment_messy(base_path, image_paths, Channels):    
    ''' 
    Obtain mean and standard deviation from the cohort
    base_path: (string) that specifies the directory where the experiment is carried out.
    images_paths: (list of strings) that specifies the names of the files executed.
    Channels: (vector of int) channels that should be included in the experiment.    
    '''
    
    # Read slide by slide
    minImage = None
    maxImage = None
    Global_hist = None
    
    # Why the heck don't we just keep track of the sum and the number of values for the sum?
    for image_path in tqdm(image_paths, desc='Calculate Mean and Standard deviation'): 
        
        # Load Image
        image = loadImage(image_path, Channels)        
                
        # To concatenate image information we sum the histograms of several images.
        if Global_hist is None:
            # What is going on here? It looks like we're introducing an upper 
            # range of intensities 10 times the max observed in the image, which is 
            # used to set the upper range of the bins for the histogram. Why?
            minImage = image.min(tuple(range(len(image.shape)-1)))
            minImage = [m*10 if m<0 else m/10 for m in minImage]
            maxImage = image.max(tuple(range(len(image.shape)-1)))
            maxImage = [m/10 if m<0 else m*10 for m in maxImage]
            # This list comprehension iterates over the channel and calculates a histogram per channel. 
            # For some weird reason the histogram is calculated based on the  image data for that channel 
            # _concatenated_ with another array containing the values 0-10.0 in increments of 1/1000000.
            # This could have a smoothing effect (having no empty bins), but still seems like an odd thing to do
            Global_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),
                                                             np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/1000000))),
                                             range=(minImage[i],maxImage[i]),bins=1000000)) 
                           for i in range(image.shape[-1])]                                    
        else:
            Local_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/1000000))),range=(minImage[i],maxImage[i]),bins=1000000)) for i in range(image.shape[-1])]                                    
            for n_g_h, g_h in enumerate(Global_hist):
                g_h[0] += Local_hist[n_g_h][0]        

    # Calculate Mean
    mean = []    
    for g_h in Global_hist:
        hist_WA = []
        den = 0
        num = 0
        for g_n, g_h_h in enumerate(g_h[0]):
            den+=(g_h_h-1)
            num+=g_h[1][g_n]*(g_h_h-1)
        mean.append(num/den)
    
    # Calculate Standard deviation
    std = []
    for hn, g_h in enumerate(Global_hist):
        hist_WA = []
        den = 0
        num = 0
        for g_n, g_h_h in enumerate(g_h[0]):
            den+=(g_h_h-1)
            num+=((g_h[1][g_n]-mean[hn])**2)*(g_h_h-1)
        std.append((num/den)**0.5)

    return np.array(mean), np.array(std)

def apply_(n_im,base_path,image_paths,Channels,mean,std,output_path,patch_size,Z_score):
    # Load Image
    im = loadImage(base_path / image_paths[n_im],Channels)        
    
    # Apply Z-score normalization
    if len(im.shape)==3 and Z_score:
        x,y,chan = im.shape[0],im.shape[1],im.shape[2] 
        im = np.reshape(im,(x*y,chan))
        im = (im-mean)/(std+1e-16)
        im = np.reshape(im,(x,y,chan))
    elif len(im.shape)==4 and Z_score:
        im = (im - np.expand_dims(np.expand_dims(np.expand_dims(mean,axis=0),axis=0),axis=0))/(np.expand_dims(np.expand_dims(np.expand_dims(std,axis=0),axis=0),axis=0)+1e-16)        

    # Save Image
    np.save(output_path+'.'.join(image_paths[n_im].split('.')[:-1])+'.npy',im)

    # Assign number of patches per image.
    return n_im, int(im.shape[0]/patch_size)*int(im.shape[1]/patch_size)

def standardize_image(work_package):
    image_path = work_package['image_path']
    output_path = work_package['output_path']
    Channels = work_package['Channels']
    patch_size = work_package['patch_size']
    Z_score = work_package['Z_score']
    mean = work_package['mean']
    std = work_package['std']
    
    im = loadImage(image_path, Channels)
    
     # Apply Z-score normalization
    if len(im.shape)==3 and Z_score:
        # expands dimensions so that broadcasting does what we want. Might want to have save the axises when creating the statistics with keepdims
        mean = mean[np.newaxis, np.newaxis, :]  
        std = std[np.newaxis, np.newaxis, :]
        centered = im - mean
        scaled = centered / std
        im = scaled
        
    elif len(im.shape)==4 and Z_score:
        raise NotImplementedError("Standardizing 4D tensors has not been reimplemented")
        im = (im - np.expand_dims(np.expand_dims(np.expand_dims(mean,axis=0),axis=0),axis=0))/(np.expand_dims(np.expand_dims(np.expand_dims(std,axis=0),axis=0),axis=0)+1e-16)        

    image_output_name = image_path.with_suffix('.npy').name
    image_output_path = output_path / image_output_name
    np.save(image_output_path, im)
    n_patches = int(im.shape[0]/patch_size)*int(im.shape[1]/patch_size)
    return image_output_path, n_patches
    

def apply_zscoreNorm(base_path,output_path,image_paths,Channels,mean,std,patch_size,z_score):        
    '''
    As the title says, apply the z-score normalization to each image, so that the global mean of each marker is 0 and the std is 1. Save the image also.
    base_path: (string) that specifies the directory where the experiment is carried out.
    output_path: (string) that specifies the directory where the images are saved.
    images_paths: (list of strings) that specifies the names of the files executed.
    Channels: (vector of int) channels that should be included in the experiment.    
    mean: (array of int) that is the mean for each marker
    std: (array of int) that is the standard deviation for each marker
    patch_size: (int) that specifies the size of the patch
    '''

    # Obtain dicts of the parallel execution
    dict_zscore = []
    [dict_zscore.append({'n_im':i,'base_path':base_path,'image_paths':image_paths,'Channels':Channels,'mean':mean,'std':std,'output_path':output_path,'patch_size':patch_size,'Z_score':z_score}) for i in range(len(image_paths))]        

    work_packages = []
    for image_path in image_paths:
        work_package = {'image_path': image_path, 'output_path': output_path, 'Channels': Channels, 'patch_size': patch_size, 'Z_score':z_score, 'mean':mean,'std':std}
        work_packages.append(work_package)

    with multiprocessing.Pool() as pool:
        num_patches_perImage_p={}
        for output_image_path, n_patches in tqdm(pool.imap_unordered(standardize_image, work_packages), total=len(work_packages), desc='Standardizing images'):
            num_patches_perImage_p[output_image_path.name] = n_patches
    #num_patches_perImage = parallel_process(dict_zscore,apply_,use_kwargs=True,n_jobs=6,front_num=0,desc='Apply Z-score normalization')
    
    # num_patches_perImage_p={}
    # for n_im, n_patches in num_patches_perImage:
    #     # Assign number of patches per image.
    #     num_patches_perImage_p['.'.join(image_paths[n_im].split('.')[:-1])+'.npy']=n_patches
    
    return num_patches_perImage_p

def extract_rois(wsi_path, output_dir):
    #mask_path = wsi_root / 'Masks'

    wsi_name, _, wsi_ext = wsi_path.name.partition('.')
    annotations_path = wsi_path.parent / f'{wsi_name}.geojson'
    
    bounding_boxes = []
    if annotations_path.exists():
        with open(annotations_path) as fp:
            annotations = geojson.load(fp)
            for feature in annotations['features']:
                geometry = feature['geometry']
                if geometry['type'] == "Polygon":
                    coordinates = geometry['coordinates'][0]  # Assume the coordinates list is a single list
                    xs, ys = zip(*coordinates)
                    min_row = int(math.floor(min(ys)))
                    min_col = int(math.floor(min(xs)))
                    max_row = int(math.ceil(max(ys)))
                    max_col = int(math.ceil(max(xs)))
                    bbox = (min_row, min_col, max_row, max_col)
                    bounding_boxes.append(bbox)
    else:
        raise RuntimeWarning(f"Image {wsi_path} has no GeoJSON annotation association with it, no regions will be extracted")
    
    # # Old code with binary masks
    # mask_path = mask_path / f'mask_{wsi_path.name}'
    # wsi_mask = tifffile.imread(mask_path)
    # label_image = measure.label(wsi_mask)  # Find connected components (white regions)
    # # Get region properties
    # regions = measure.regionprops(label_image)  
    # # Extract bounding boxes
    # bounding_boxes = [region.bbox for region in regions]

    # # We could use pyvips to only read the regions, but for now we'll do the resource intensive thing and read the whole image
    #wsi = tifffile.imread(wsi_path)
    region_images = []
    with tifffile.TiffFile(wsi_path) as tif:
    # `bounding_boxes` contains (min_row, min_col, max_row, max_col) tuples
        wsi = tif.asarray(out='memmap')
        wsi_name, _, wsi_ext = wsi_path.name.partition('.')
            
        for bbox in tqdm(bounding_boxes, desc="Extracting ROIs"):
            min_row, min_col, max_row, max_col = bbox
            coords_string = f'({min_row},{min_col}),({max_row}, {max_col})'
            image_region_name = f'{wsi_name}[{coords_string}].{wsi_ext}'
            output_path = output_dir / image_region_name
            
            if not output_path.exists():
                image_region = wsi[min_row:max_row, min_col:max_col]
                # Do stuff with the metadata
                # tifffile seems to assume channel is first, let's transpose 
                # it and see if it does what we want
                image_region = image_region.transpose((2,0,1))
                tifffile.imwrite(output_path, image_region, metadata={'axes': 'CYX'})
            region_images.append(output_path)
    return region_images
            
        
def get_image_metadata(metadata_path):    
    metadata = pd.read_excel(metadata_path, engine='openpyxl')
    
    image_names_to_columns = dict()
    for row in metadata.to_dict('records'):
        image_name = row['Image_Names']
        other_cols = {col: val for col, val in row.items() if col != 'Image_Names'}
        image_names_to_columns[image_name] = other_cols
    return image_names_to_columns


def preprocess_wsis(path: str, image_types=['tif', 'tiff', 'npy']):
    path = Path(path)
    # Paths                            
    base_path = path / 'Raw_Data/'
    wsi_root_path = base_path / 'WSI'
    
    
    wsi_image_dir = wsi_root_path / 'Images'
    wsi_metadata_path = wsi_root_path / 'Experiment_Information'/ 'Image_Labels.xlsx'
    wsi_image_names_to_columns = get_image_metadata(wsi_metadata_path)
    
    region_output_dir = base_path / 'Images'
    region_output_dir.mkdir(exist_ok=True, parents=True)
    experiment_info_output_dir = base_path / 'Experiment_Information'
    experiment_info_output_dir.mkdir(exist_ok=True, parents=True)
    output_metadata = experiment_info_output_dir / 'Image_Labels.xlsx'
    #if output_metadata.exists():
    #    output_image_names_to_columns = get_image_metadata(output_metadata)
    #else:
    output_image_names_to_columns = dict()
            
    wsi_paths = set()
    for image_type in image_types:
        wsis_of_type = wsi_image_dir.glob(f'*.{image_type}')
        wsi_paths.update(wsis_of_type)
    
    image_to_patient_records = []
    for wsi_path in tqdm(wsi_paths, desc="Processing WSI"):
        wsi_name = wsi_path.name
        image_metadata = wsi_image_names_to_columns[wsi_name]
        image_metadata['Subject_Names'] = wsi_name
        output_image_names_to_columns[wsi_name] = image_metadata
        
        roi_paths = extract_rois(wsi_path, region_output_dir)
        
        for roi_path in roi_paths:
            roi_name = roi_path.name
            image_to_patient_records.append({'Image_Name': roi_name, 'Subject_Name': wsi_name})
            # Don't overwrite existing information (in case manually edited)
    
    output_records = []
    
    for name, columns in output_image_names_to_columns.items():
        output_record = {'Image_Names': name}
        output_record.update(columns)
        output_records.append(output_record)
    
    updated_output_metadata = pd.DataFrame.from_records(output_records)
    updated_output_metadata.to_excel(output_metadata, index=False)
    
    image_to_patient_output = experiment_info_output_dir / 'Patient_to_Image.xlsx'
    image_to_subject_df = pd.DataFrame.from_records(image_to_patient_records)
    image_to_subject_df.to_excel(image_to_patient_output, index=False)
    
    output_channels_file = experiment_info_output_dir / 'Channels.txt'
    if not output_channels_file.exists():
        wsi_channels_file = wsi_root_path / 'Experiment_Information' / 'Channels.txt'
        shutil.copy(wsi_channels_file, output_channels_file)


def preprocess_images(path,ZScoreNormalization,patch_size, image_types=['tif', 'tiff', 'npy']):
    '''
    path: (path) where is the experiment to execute
    ZScoreNormalization: (boolean) Whether to normalize the images or not.
    Images_Names_Ends_In: (string) that specifies the image type    
    patch_size: (int) that specifies the size of the patch
    '''
    path = Path(path)
    base_path = path / "Raw_Data"
    images_path =  base_path / 'Images'
    pcl_path = path / 'Patch_Contrastive_Learning'
    output_path = pcl_path / 'Preprocessed_Images/'
    
    # Create dir
    pcl_path.mkdir(exist_ok=True, parents=True)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Obtain Image Paths. Don't assume all of them are images. 
    image_paths = set()
    for image_type in image_types:
        type_paths = images_path.glob(f'*.{image_type}')
        image_paths.update(type_paths)
        
    image_paths = sorted(image_paths)
    preprocessed_paths = sorted(output_path.glob('*.npy'))
    z_score = True # Apply z-score normalization or not

    Channels, Marker_Names = utilz.load_channels(str(base_path))
    
    
    # to_preprocess = set([image_path.with_suffix('').name for image_path in image_paths])
    # for preprocessed_path in preprocessed_paths:
    #     file_name = preprocessed_path.with_suffix('').name
    #     if file_name in to_preprocess:
    #         to_preprocess.remove(file_name)
    
    # Iterate Images to obtain mean and std of marker distribution.
    #random.shuffle(image_paths)  # Don't see why we would need to shuffle here.
    
    print('Preprocess a cohort of ', str(len(image_paths)),' subjects:')
    mean, std = Mean_std_experiment(base_path=base_path / 'Images/', image_paths=image_paths,Channels=Channels)

    # Apply z-score normalization and save them to efficient structures.
    num_patches_perImage = apply_zscoreNorm(base_path / 'Images/', output_path, image_paths, Channels, mean, std, patch_size, z_score)        
    
    # Write num_patches_perImage to csv.
    with open(output_path / "Num_patches_perImage.csv", "w") as fp:
        w = csv.writer(fp)
        for key, val in num_patches_perImage.items():
            w.writerow([key, val])
    # else:       
    #     # What the heck is this? 
    #     with tqdm(total=len(image_paths), ascii=True, desc='Calculate Mean and Standard deviation') as bar_folds:            
    #         bar_folds.update(len(image_paths))
    #     with tqdm(total=len(image_paths), ascii=True, desc='Apply Z-score normalization') as bar_folds:            
    #         bar_folds.update(len(image_paths))
