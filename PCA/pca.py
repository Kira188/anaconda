import cv2
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
image = cv2.imread('/Users/kiranseenivasan/Documents/anaconda/example_image.jpg', cv2.IMREAD_GRAYSCALE)

def compress_image(n_components, image):
    pca = PCA(n_components=n_components)
    image_compressed = pca.fit_transform(image)
    return pca.inverse_transform(image_compressed)

#number of principle components
k=10
# Compress images with different numbers of principal components
reconstructed_image = compress_image(k, image)
#put image
cv2.imshow('Original Image', image)
cv2.imwrite('Original.jpg', image)
cv2.imwrite('Recon.jpg', reconstructed_image.astype(np.uint8))
cv2.imshow('Reconstructed Image', reconstructed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()