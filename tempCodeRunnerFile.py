import numpy as np
import cv2

# Load image
image = cv2.imread('example_image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Singular Value Decomposition
U, S, V = np.linalg.svd(image, full_matrices=False)

# Choose the number of singular values to keep
k=100

print(U.view())
print(S.view())


# Reconstruct the image using the selected singular values
reconstructed_image = np.dot(U[:, :k] * S[:k], V[:k, :])

# Display the original and reconstructed images
cv2.imshow('Original Image', image)
cv2.imwrite('Original.jpg', image)
cv2.imwrite('Recon.jpg', reconstructed_image.astype(np.uint8))
cv2.imshow('Reconstructed Image', reconstructed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

