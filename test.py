import os
import numpy as np
import logging
import imageio
from skimage.morphology import remove_small_objects
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d

def segment_images(inpDir, outDir, config_data):
    """ Workflow for data with a spotty appearance
    in each 2d frame such as fibrillarin and beta catenin.

    Args:
        inpDir : path to the input directory
        outDir : path to the output directory
        config_data : configuration data
    """

    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    inpDir_files = os.listdir(inpDir)
    for i, f in enumerate(inpDir_files):
        logger.info('Segmenting image : {}'.format(f))

        # Load image
        img_path = os.path.join(inpDir, f)
        image = imageio.v2.imread(img_path)
        if image.ndim == 3:
            image = image[:, :, 0]  # Take the first channel if it's a color image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Make it 3D with one slice
        struct_img0 = image.astype(np.float32)

        # Main algorithm
        intensity_scaling_param = config_data['intensity_scaling_param']
        struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param) 
        gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma'] 
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        s2_param = [config_data['s2_param']]  # Ensure it's in the correct format
        bw = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)
        minArea = config_data['minArea']
        seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
        seg = seg > 0
        out_img = seg.astype(np.uint8)
        out_img[out_img > 0] = 255

        # Save output image
        out_img_path = os.path.join(outDir, f)
        imageio.imwrite(out_img_path, out_img.squeeze())  # Squeeze to remove single-dimensional entries

# Example usage
config_data = {
    'intensity_scaling_param': [0.0, 100.0],
    'gaussian_smoothing_sigma': 1.0,
    's2_param': [1.0, 1.0],  # Example parameter values
    'minArea': 10
}
inpDir = '/Users/Gabrielle/Desktop/Test'
outDir = '/Users/Gabrielle/Desktop/cnn'

segment_images(inpDir, outDir, config_data)
