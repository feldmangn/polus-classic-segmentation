import numpy as np
import math
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QComboBox
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_3d,
    image_smoothing_gaussian_slice_by_slice,
    edge_preserving_smoothing_3d
)
from skimage.morphology import remove_small_objects, dilation, ball, disk
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Conv2DTranspose, Input, Concatenate
import tensorflow as tf
from pathlib import Path
import imageio
import cv2

# Attempt to import from ome_types
try:
    from ome_types.model import OME, Pixels, Channel, Image
except ImportError:
    print("Warning: Failed to import OME, Pixels, and Channel from ome_types.model. Using fallback implementation.")

class SegmentWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.img_layer = None

        # Algorithm selection
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItem("CurvyLinear")
        self.algorithm_selector.addItem("Dots")
        self.algorithm_selector.addItem("Filament3D")
        self.algorithm_selector.addItem("Spotty")
        self.algorithm_selector.addItem("CellNuclei")
        self.algorithm_selector.currentIndexChanged.connect(self.update_parameters_visibility)

        # Common parameters
        self.intensity_scaling_param_min_spinbox = QDoubleSpinBox()
        self.intensity_scaling_param_min_spinbox.setMinimum(0.0)
        self.intensity_scaling_param_min_spinbox.setMaximum(100.0)
        self.intensity_scaling_param_min_spinbox.setValue(0.0)

        self.intensity_scaling_param_max_spinbox = QDoubleSpinBox()
        self.intensity_scaling_param_max_spinbox.setMinimum(0.0)
        self.intensity_scaling_param_max_spinbox.setMaximum(100.0)
        self.intensity_scaling_param_max_spinbox.setValue(100.0)

        self.gaussian_smoothing_sigma_spinbox = QDoubleSpinBox()
        self.gaussian_smoothing_sigma_spinbox.setMinimum(0.0)
        self.gaussian_smoothing_sigma_spinbox.setMaximum(10.0)
        self.gaussian_smoothing_sigma_spinbox.setValue(1.0)

        self.minArea_spinbox = QSpinBox()
        self.minArea_spinbox.setMinimum(0)
        self.minArea_spinbox.setMaximum(1000)
        self.minArea_spinbox.setValue(10)

        # CurvyLinear specific parameters
        self.f2_param_first_label = QLabel("F2 Param First (CurvyLinear)")
        self.f2_param_first_spinbox = QDoubleSpinBox()
        self.f2_param_first_spinbox.setMinimum(0.0)
        self.f2_param_first_spinbox.setMaximum(10.0)
        self.f2_param_first_spinbox.setValue(1.0)

        self.f2_param_second_label = QLabel("F2 Param Second (CurvyLinear)")
        self.f2_param_second_spinbox = QDoubleSpinBox()
        self.f2_param_second_spinbox.setMinimum(0.0)
        self.f2_param_second_spinbox.setMaximum(10.0)
        self.f2_param_second_spinbox.setValue(0.5)

        # Dots specific parameters
        self.s3_param_sigma_label = QLabel("S3 Param Sigma (Dots)")
        self.s3_param_sigma_spinbox = QDoubleSpinBox()
        self.s3_param_sigma_spinbox.setMinimum(0.0)
        self.s3_param_sigma_spinbox.setMaximum(10.0)
        self.s3_param_sigma_spinbox.setValue(1.0)

        self.s3_param_threshold_label = QLabel("S3 Param Threshold (Dots)")
        self.s3_param_threshold_spinbox = QDoubleSpinBox()
        self.s3_param_threshold_spinbox.setMinimum(0.0)
        self.s3_param_threshold_spinbox.setMaximum(10.0)
        self.s3_param_threshold_spinbox.setValue(0.5)

        # Filament3D specific parameters
        self.f3_param_label = QLabel("F3 Param (Filament3D)")
        self.f3_param_spinbox = QDoubleSpinBox()
        self.f3_param_spinbox.setMinimum(0.0)
        self.f3_param_spinbox.setMaximum(10.0)
        self.f3_param_spinbox.setValue(1.0)

        self.preprocessing_function_label = QLabel("Preprocessing Function (Filament3D)")
        self.preprocessing_function_selector = QComboBox()
        self.preprocessing_function_selector.addItem("image_smoothing_gaussian_3d")
        self.preprocessing_function_selector.addItem("edge_preserving_smoothing_3d")

        # Spotty specific parameters
        self.s2_param_label = QLabel("S2 Param (Spotty)")
        self.s2_param_spinbox = QDoubleSpinBox()
        self.s2_param_spinbox.setMinimum(0.0)
        self.s2_param_spinbox.setMaximum(10.0)
        self.s2_param_spinbox.setValue(1.0)

        # Create button
        self.segment_button = QPushButton("Segment Image")
        self.segment_button.clicked.connect(self.segment_image)

        # Create refresh button
        self.refresh_button = QPushButton("Refresh Image Layer")
        self.refresh_button.clicked.connect(self.update_image_layer)

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("Algorithm"))
        self.layout.addWidget(self.algorithm_selector)
        self.layout.addWidget(QLabel("Intensity Scaling Param Min"))
        self.layout.addWidget(self.intensity_scaling_param_min_spinbox)
        self.layout.addWidget(QLabel("Intensity Scaling Param Max"))
        self.layout.addWidget(self.intensity_scaling_param_max_spinbox)
        self.layout.addWidget(QLabel("Gaussian Smoothing Sigma"))
        self.layout.addWidget(self.gaussian_smoothing_sigma_spinbox)
        self.layout.addWidget(QLabel("Min Area"))
        self.layout.addWidget(self.minArea_spinbox)

        # CurvyLinear parameters
        self.layout.addWidget(self.f2_param_first_label)
        self.layout.addWidget(self.f2_param_first_spinbox)
        self.layout.addWidget(self.f2_param_second_label)
        self.layout.addWidget(self.f2_param_second_spinbox)

        # Dots parameters
        self.layout.addWidget(self.s3_param_sigma_label)
        self.layout.addWidget(self.s3_param_sigma_spinbox)
        self.layout.addWidget(self.s3_param_threshold_label)
        self.layout.addWidget(self.s3_param_threshold_spinbox)

        # Filament3D parameters
        self.layout.addWidget(self.f3_param_label)
        self.layout.addWidget(self.f3_param_spinbox)
        self.layout.addWidget(self.preprocessing_function_label)
        self.layout.addWidget(self.preprocessing_function_selector)

        # Spotty parameters
        self.layout.addWidget(self.s2_param_label)
        self.layout.addWidget(self.s2_param_spinbox)

        self.layout.addWidget(self.segment_button)
        self.layout.addWidget(self.refresh_button)

        self.setLayout(self.layout)
        
        # Connect the layer change event to the update function
        self.viewer.layers.events.changed.connect(self.update_image_layer)

        # Manually set the image layer if one exists
        self.update_image_layer(None)
        
        # Initial parameter visibility update
        self.update_parameters_visibility()

    def update_parameters_visibility(self):
        algorithm = self.algorithm_selector.currentText()
        is_curvy_linear = algorithm == "CurvyLinear"
        is_dots = algorithm == "Dots"
        is_filament3d = algorithm == "Filament3D"
        is_spotty = algorithm == "Spotty"
        is_cell_nuclei = algorithm == "CellNuclei"

        # Update visibility of common parameters
        show_common_params = not is_cell_nuclei
        self.intensity_scaling_param_min_spinbox.setVisible(show_common_params)
        self.intensity_scaling_param_max_spinbox.setVisible(show_common_params)
        self.gaussian_smoothing_sigma_spinbox.setVisible(show_common_params)
        self.minArea_spinbox.setVisible(show_common_params)

        # Update visibility of CurvyLinear parameters
        self.f2_param_first_label.setVisible(is_curvy_linear)
        self.f2_param_first_spinbox.setVisible(is_curvy_linear)
        self.f2_param_second_label.setVisible(is_curvy_linear)
        self.f2_param_second_spinbox.setVisible(is_curvy_linear)

        # Update visibility of Dots parameters
        self.s3_param_sigma_label.setVisible(is_dots)
        self.s3_param_sigma_spinbox.setVisible(is_dots)
        self.s3_param_threshold_label.setVisible(is_dots)
        self.s3_param_threshold_spinbox.setVisible(is_dots)

        # Update visibility of Filament3D parameters
        self.f3_param_label.setVisible(is_filament3d)
        self.f3_param_spinbox.setVisible(is_filament3d)
        self.preprocessing_function_label.setVisible(is_filament3d)
        self.preprocessing_function_selector.setVisible(is_filament3d)

        # Update visibility of Spotty parameters
        self.s2_param_label.setVisible(is_spotty)
        self.s2_param_spinbox.setVisible(is_spotty)

    def update_image_layer(self, event):
        # Check if there are any image layers and set the latest one
        print("Checking layers in the viewer...")
        found = False
        for layer in self.viewer.layers:
            print("Layer:", layer, "Type:", type(layer))
            if isinstance(layer, napari.layers.Image):
                self.set_image_layer(layer)
                found = True
                break
        if not found:
            print("No image layer found.")
        print("Update image layer called. Current image layer:", self.img_layer)

    def set_image_layer(self, img_layer):
        self.img_layer = img_layer
        print("Image layer set to:", self.img_layer)

    def segment_image(self):
        if self.img_layer is None:
            print("No image layer set!")
            return

        print("Segmenting image...")

        # Get selected algorithm
        algorithm = self.algorithm_selector.currentText()
        print("Selected algorithm:", algorithm)

        # Preprocessing
        image_data = self.img_layer.data
        print("Image data shape:", image_data.shape)
        struct_img0 = image_data.astype(np.float32)

        if algorithm != "CellNuclei":
            intensity_scaling_param_min = self.intensity_scaling_param_min_spinbox.value()
            intensity_scaling_param_max = self.intensity_scaling_param_max_spinbox.value()
            gaussian_smoothing_sigma = self.gaussian_smoothing_sigma_spinbox.value()
            minArea = self.minArea_spinbox.value()

            if intensity_scaling_param_max == 0:
                struct_img = intensity_normalization(struct_img0, scaling_param=[intensity_scaling_param_min])
            elif intensity_scaling_param_max > 0:
                struct_img = intensity_normalization(struct_img0, scaling_param=[intensity_scaling_param_min, intensity_scaling_param_max])

            if len(struct_img.shape) == 3:
                structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
            else:
                structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
            
            print("Smoothed image shape:", structure_img_smooth.shape)
        else:
            struct_img = struct_img0 / 255.0
            structure_img_smooth = struct_img
            minArea = None

        if algorithm == "CurvyLinear":
            f2_param_first = self.f2_param_first_spinbox.value()
            f2_param_second = self.f2_param_second_spinbox.value()
            f2_param = [[f2_param_first, f2_param_second]]
            bw = filament_2d_wrapper(structure_img_smooth, f2_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
        
        elif algorithm == "Dots":
            s3_param_sigma = self.s3_param_sigma_spinbox.value()
            s3_param_threshold = self.s3_param_threshold_spinbox.value()
            s3_param = [[s3_param_sigma, s3_param_threshold]]
            bw = dot_3d_wrapper(structure_img_smooth, s3_param)
            Mask = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
            local_max = peak_local_max(struct_img, labels=label(Mask), min_distance=2, footprint=np.ones((3, 3)))
            local_max_mask = np.zeros_like(struct_img, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            footprint = ball(1) if struct_img.ndim == 3 else disk(1)
            Seed = dilation(local_max_mask, footprint=footprint)
            Watershed_Map = -1 * distance_transform_edt(bw)
            seg = watershed(Watershed_Map, markers=label(Seed), mask=Mask, watershed_line=True)
            seg = remove_small_objects(seg > 0, min_size=minArea, connectivity=1)
            seg = seg > 0

        elif algorithm == "Filament3D":
            f3_param = self.f3_param_spinbox.value()
            preprocessing_function = self.preprocessing_function_selector.currentText()
            if preprocessing_function == 'image_smoothing_gaussian_3d':
                structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
            elif preprocessing_function == 'edge_preserving_smoothing_3d':
                structure_img_smooth = edge_preserving_smoothing_3d(struct_img)
            f3_param = [f3_param]
            bw = filament_3d_wrapper(structure_img_smooth, f3_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
            seg = seg > 0

        elif algorithm == "Spotty":
            s2_param = self.s2_param_spinbox.value()
            s2_param = [[s2_param, s2_param]]
            bw = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
            seg = seg > 0
        
        elif algorithm == "CellNuclei":
            self.segment_cell_nuclei(struct_img0)
            return

        # Ensure the output is correctly formatted
        labeled_image = seg.astype(np.uint32)

        print("Segmentation completed.")

        # Display or use the segmentation result as needed
        self.viewer.add_labels(labeled_image)

    def segment_cell_nuclei(self, image):
        def unet(in_shape=(256,256,3), alpha=0.1, dropout=None):
            Unet_Input = Input(shape=in_shape)
            conv1_1 = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(Unet_Input)
            relu1_1 = LeakyReLU(alpha = alpha)(conv1_1)
            conv1_2 = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu1_1)
            relu1_2 = LeakyReLU(alpha = alpha)(conv1_2)    
            bn1 = BatchNormalization()(relu1_2)
            maxpool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn1)    
            conv2_1 = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool1)
            relu2_1 = LeakyReLU(alpha = alpha)(conv2_1)    
            conv2_2 = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu2_1)
            relu2_2 = LeakyReLU(alpha = alpha)(conv2_2)    
            bn2 = BatchNormalization()(relu2_2)
            maxpool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn2)    
            conv3_1 = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool2)
            relu3_1 = LeakyReLU(alpha = alpha)(conv3_1)    
            conv3_2 = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu3_1)
            relu3_2 = LeakyReLU(alpha = alpha)(conv3_2)    
            bn3 = BatchNormalization()(relu3_2)
            maxpool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn3)    
            conv4_1 = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool3)
            relu4_1 = LeakyReLU(alpha = alpha)(conv4_1)    
            conv4_2 = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu4_1)
            relu4_2 = LeakyReLU(alpha = alpha)(conv4_2)    
            bn4 = BatchNormalization()(relu4_2)
            maxpool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn4)        
            conv5_1 = Conv2DTranspose(256, kernel_size=(3,3), strides = (2,2), padding = 'same')(maxpool4)
            relu5_1 = LeakyReLU(alpha = alpha)(conv5_1)
            conc5 = Concatenate(axis=3)([relu5_1, relu4_2])
            conv5_2 = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc5)
            relu5_2 = LeakyReLU(alpha = alpha)(conv5_2)
            bn5 = BatchNormalization()(relu5_2)
            conv6_1 = Conv2DTranspose(128, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn5)
            relu6_1 = LeakyReLU(alpha = alpha)(conv6_1)
            conc6 = Concatenate(axis=3)([relu6_1, relu3_2])
            conv6_2 = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc6)
            relu6_2 = LeakyReLU(alpha = alpha)(conv6_2)
            bn6 = BatchNormalization()(relu6_2)
            conv7_1 = Conv2DTranspose(64, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn6)
            relu7_1 = LeakyReLU(alpha = alpha)(conv7_1)
            conc7 = Concatenate(axis=3)([relu7_1, relu2_2])
            conv7_2 = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc7)
            relu7_2 = LeakyReLU(alpha = alpha)(conv7_2)
            bn7 = BatchNormalization()(relu7_2)
            conv8_1 = Conv2DTranspose(32, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn7)
            relu8_1 = LeakyReLU(alpha = alpha)(conv8_1)
            conc8 = Concatenate(axis=3)([relu8_1, relu1_2])
            conv8_2 = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc8)
            relu8_2 = LeakyReLU(alpha = alpha)(conv8_2)
            Unet_Output = Conv2D(1, kernel_size=(1,1), strides = (1,1), padding='same', activation='sigmoid')(relu8_2)
            Unet = Model(Unet_Input, Unet_Output)
            return Unet

        model = unet()
        model.load_weights('/Users/Gabrielle/polus-classic-segmentation/unet.h5')  # Adjust the path to your model

        def padding(image):
            row, col, _ = image.shape
            m, n = math.ceil(row/256), math.ceil(col/256)
            required_rows, required_cols = m*256, n*256  
            top = (required_rows - row) // 2
            bottom = required_rows - row - top
            left = (required_cols - col) // 2
            right = required_cols - col - left
            pad_dimensions = (top, bottom, left, right)
            final_image = np.zeros((required_rows, required_cols, 3))
            for i in range(3):
                final_image[:,:,i] = cv2.copyMakeBorder(image[:,:,i], top, bottom, left, right, cv2.BORDER_REFLECT)
            return final_image, pad_dimensions

        img = np.dstack((image, image, image))
        padded_img, pad_dimensions = padding(img)
        print(f"Padded image shape: {padded_img.shape}")
        final_img = np.zeros((padded_img.shape[0], padded_img.shape[1]))
        for i in range(int(padded_img.shape[0]/256)):
            for j in range(int(padded_img.shape[1]/256)):
                temp_img = padded_img[i*256:(i+1)*256, j*256:(j+1)*256]
                inp = np.expand_dims(temp_img, axis=0)
                x = model.predict(inp)
                out = x[0,:,:,0]
                final_img[i*256:(i+1)*256, j*256:(j+1)*256] = out 
        top_pad, bottom_pad, left_pad, right_pad = pad_dimensions
        out_image = final_img[top_pad:final_img.shape[0]-bottom_pad, left_pad:final_img.shape[1]-right_pad]
        print(f"Final output image shape: {out_image.shape}, unique values: {np.unique(out_image)}")
        out_image = np.rint(out_image) * 255
        out_image = out_image.astype(np.uint8)

        # Display the result in napari
        self.viewer.add_labels(out_image)

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(SegmentWidget(viewer))
    napari.run()
