import numpy as np
import cv2
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, edge_preserving_smoothing_3d
from skimage.morphology import remove_small_objects
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
import numpy as np
from napari import Viewer



def segment_image(image: np.ndarray, intensity_scaling_param: list, gaussian_smoothing_sigma: float, f2_param: list, minArea: int, preprocessing_function: str) -> np.ndarray:
        structure_channel = 0
        struct_img0 = image[..., structure_channel]
        struct_img0 = struct_img0.astype(np.float32)

        # Main algorithm
        if intensity_scaling_param[1] == 0:
            struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param[:1])
        else:
            struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
        
        if preprocessing_function == 'image_smoothing_gaussian_3d':
            structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        elif preprocessing_function == 'edge_preserving_smoothing_3d':
            structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
        
        bw = filament_2d_wrapper(structure_img_smooth, f2_param)
        
        seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
        seg = seg > 0

        out_img = seg.astype(np.uint8) * 255
        
        return out_img

# Define your image data
image = cv2.imread('/Users/Gabrielle/Desktop/cameraman.png')

# Define your parameter values
# Corrected f2_param to be a list of lists with two elements each
f2_param = [[1.0, 0.5]]
intensity_scaling_param = [0, 100]
gaussian_smoothing_sigma = 2.0
minArea = 10
preprocessing_function = 'image_smoothing_gaussian_3d'

# Call the segment_image function
segmented_image = segment_image(image, intensity_scaling_param, gaussian_smoothing_sigma, f2_param, minArea, preprocessing_function)

# Now you have your segmented image
import matplotlib.pyplot as plt

# Plot both images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
