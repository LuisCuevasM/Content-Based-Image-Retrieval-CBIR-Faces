
from skimage.feature import hog
from skimage.io import imread


def  handcrafted_method(path,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(2, 2)):
 img = imread(path)
 fd, hog_image = hog(img, orientations, pixels_per_cell,
                     cells_per_block, block_norm='L2-Hys',
                     visualize=True, channel_axis=-1)
 return fd