# Import important libraries
import cv2
import numpy as np

# Load the image
imgpath = "4.2.07.tiff"
img = cv2.imread(imgpath, 0)

# Calculating the mean columnwise
M = np.mean(img.T, axis=1)

# Sustracting the mean columnwise
C = img - M

# Calculating the covariance matrix
V = np.cov(C.T)

# Computing the eigenvalues and eigenvectors of covarince matrix
values, vectors = np.linalg.eig(V)

p = np.size(vectors, axis =1)

# Sorting the eigen values in ascending order
idx = np.argsort(values)
idx = idx[::-1]

# Sorting eigen vectors
vectors = vectors[:,idx]
values = values[idx]

# PCs used for reconstruction (can be varied)
num_PC = 200

# Cutting the PCs
if num_PC <p or num_PC >0:
	vectors = vectors[:, range(num_PC)]

# Reconstructing the image with PCs
score = np.dot(vectors.T, C)
constructed_img = np.dot(vectors, score) + M
constructed_img = np.uint8(np.absolute(constructed_img))

# Show reconstructed image
cv2.imshow("Reconstructed Image", constructed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()