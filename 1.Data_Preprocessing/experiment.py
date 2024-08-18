import cv2
from skimage import io
from skimage import img_as_float
from  matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from  skimage.filters import sobel

my_image=io.imread('images/masked_images/1.png')

print(type(my_image)) # image is a numpy array

print(my_image.dtype) # image is of dtype

print(my_image.shape) # image is of shape

# open ImageJ and see the locations of pixels to verify the pixel position.
# (x,y) as spatial coordinate corresponds to the numpy array position as [y,x]

print(my_image[338][578]) # image pixel position (578,338) in image has value of RGB as [128,128,0] -- yellow color

my_image=cv2.resize(my_image,(512,533))
#cv2.imshow('image',my_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
my_float_image=img_as_float(my_image)
print(my_float_image.min(),my_float_image.max())
print(my_image.min(),my_image.max())
# random_image=np.random.random([500,500]) # random noise for values between 0 and 1
# print(random_image)
# cv2.imshow("pic",random_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################################################
my_image=io.imread('images/original_images/1.png')
print(my_image.shape)
img2=sobel(my_image)
# cv2.imshow("edge",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Edge detection for the default image

image = camera()
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

# plt.tight_layout()
# plt.show()

#################################################################################

