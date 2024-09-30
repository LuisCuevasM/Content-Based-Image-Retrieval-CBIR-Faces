
from skimage.feature import hog
from skimage.io import imread


def  handcrafted_method(path):
 img = imread(path)
 fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm='L2-Hys',
                     visualize=True, channel_axis=-1)
 return fd


print(handcrafted_method(
    r"C:\Users\amand\Downloads\CBIR_Faces_Dataset_2024\CBIR Faces Dataset 2024\n000493\n000493_2.jpg"))