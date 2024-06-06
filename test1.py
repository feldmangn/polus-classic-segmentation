import napari
from skimage import io

# Load an image from file
data = io.imread('/Users/Gabrielle/Desktop/cameraman.png')

# Initialize Napari viewer
viewer = napari.Viewer()

# Add the image data as an image layer
viewer.add_image(data, name='example_image')

# Start the Napari event loop
napari.run()