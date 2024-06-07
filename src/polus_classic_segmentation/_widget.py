import numpy as np
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

        # Get common parameter values from the widgets
        intensity_scaling_param_min = self.intensity_scaling_param_min_spinbox.value()
        intensity_scaling_param_max = self.intensity_scaling_param_max_spinbox.value()
        gaussian_smoothing_sigma = self.gaussian_smoothing_sigma_spinbox.value()
        minArea = self.minArea_spinbox.value()

        # Preprocessing
        image_data = self.img_layer.data
        print("Image data shape:", image_data.shape)
        struct_img0 = image_data.astype(np.float32)

        if intensity_scaling_param_max == 0:
            struct_img = intensity_normalization(struct_img0, scaling_param=[intensity_scaling_param_min])
        elif intensity_scaling_param_max > 0:
            struct_img = intensity_normalization(struct_img0, scaling_param=[intensity_scaling_param_min, intensity_scaling_param_max])

        if len(struct_img.shape) == 3:
            structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
        else:
            structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        
        print("Smoothed image shape:", structure_img_smooth.shape)

        if algorithm == "CurvyLinear":
            # Get CurvyLinear specific parameters
            f2_param_first = self.f2_param_first_spinbox.value()
            f2_param_second = self.f2_param_second_spinbox.value()
            f2_param = [[f2_param_first, f2_param_second]]

            # Thresholding and Segmentation for CurvyLinear
            bw = filament_2d_wrapper(structure_img_smooth, f2_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
        
        elif algorithm == "Dots":
            # Get Dots specific parameters
            s3_param_sigma = self.s3_param_sigma_spinbox.value()
            s3_param_threshold = self.s3_param_threshold_spinbox.value()
            s3_param = [[s3_param_sigma, s3_param_threshold]]

            # Thresholding and Segmentation for Dots
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
            # Get Filament3D specific parameters
            f3_param = self.f3_param_spinbox.value()
            preprocessing_function = self.preprocessing_function_selector.currentText()

            # Preprocessing for Filament3D
            if preprocessing_function == 'image_smoothing_gaussian_3d':
                structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
            elif preprocessing_function == 'edge_preserving_smoothing_3d':
                structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

            # Segmentation for Filament3D
            f3_param = [f3_param]  # Ensure it's in a list
            bw = filament_3d_wrapper(structure_img_smooth, f3_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
            seg = seg > 0

        elif algorithm == "Spotty":
            # Get Spotty specific parameters
            s2_param = self.s2_param_spinbox.value()
            s2_param = [[s2_param, s2_param]]  # Ensure it's in the correct format

            # Thresholding and Segmentation for Spotty
            bw = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)
            seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
            seg = seg > 0

        # Ensure the output is correctly formatted
        labeled_image = seg.astype(np.uint32)

        print("Segmentation completed.")

        # Display or use the segmentation result as needed
        self.viewer.add_labels(labeled_image)

# Function to return the SegmentWidget instance as a QWidget
def create_segment_widget(viewer):
    return SegmentWidget(viewer)

# Create the Napari viewer
viewer = napari.Viewer()

# Add the segment widget to Napari viewer
widget = create_segment_widget(viewer)
viewer.window.add_dock_widget(widget)

# Run Napari
napari.run()
