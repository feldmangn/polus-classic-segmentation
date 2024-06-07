import numpy as np
import h5py
import logging
import os
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, dilation, ball
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.io import imsave

def segment_image_3d(image: np.ndarray, config_data: dict) -> np.ndarray:
    """ Main segmentation algorithm for 3D images """

    struct_img0 = image.astype(np.float32)

    # Intensity normalization
    intensity_scaling_param = config_data['intensity_scaling_param']
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
        
    # Gaussian smoothing
    gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma']
    if config_data["gaussian_smoothing"] == "gaussian_slice_by_slice":
        structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
    else:
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)    
    
    # Dot segmentation
    s3_param = config_data['s3_param']
    bw = dot_3d_wrapper(structure_img_smooth, s3_param)
    
    # Post-processing
    minArea = config_data['minArea']
    Mask = remove_small_objects(bw > 0, min_size=minArea, connectivity=1, in_place=False)
    Seed = dilation(peak_local_max(struct_img, labels=label(Mask), min_distance=2, indices=False), selem=ball(1))
    Watershed_Map = -1 * distance_transform_edt(bw)
    seg = watershed(Watershed_Map, label(Seed), mask=Mask, watershed_line=True)
    seg = remove_small_objects(seg > 0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg > 0

    out_img = seg.astype(np.uint8)
    out_img[out_img > 0] = 255

    return out_img

def save_image_as_hdf5(image: np.ndarray, file_path: str):
    """ Save the image as an HDF5 file """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('image', data=image)

# Define your input 3D image data (path to the previously generated HDF5 file)
input_path = '/Users/Gabrielle/Downloads/test_3d_image.h5'

# Define your parameter values
config_data = {
    'intensity_scaling_param': [0, 100],
    'gaussian_smoothing_sigma': 2.0,
    'gaussian_smoothing': 'gaussian_3d',  # or 'gaussian_slice_by_slice'
    's3_param': [[1.0, 0.5]],  # Adjust as needed
    'minArea': 10
}

# Load the 3D image from the HDF5 file
with h5py.File(input_path, 'r') as f:
    image = f['image'][:]

# Perform segmentation
segmented_image = segment_image_3d(image, config_data)

# Save the segmented image to an HDF5 file
output_path = '/mnt/data/segmented_3d_image.h5'
save_image_as_hdf5(segmented_image, output_path)

output_path
