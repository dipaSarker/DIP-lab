import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load Grayscale Image
# Replace 'image.jpg' with your grayscale image file path
image = cv2.imread('image.tif', 0)
if image is None:
    raise FileNotFoundError("Image not found. Please provide a valid file path.")

# Step 2: Manually Calculate Histogram
def calculate_histogram(image):
    histogram = np.zeros(256, dtype=int)
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

# Step 3: Calculate PDF
def calculate_pdf(histogram, total_pixels):
    pdf = np.zeros_like(histogram, dtype=float)
    for i in range(len(histogram)):
        pdf[i] = histogram[i] / total_pixels
    return pdf

# Step 4: Calculate CDF
def calculate_cdf(pdf):
    cdf = np.zeros_like(pdf, dtype=float)
    cumulative_sum = 0
    for i in range(len(pdf)):
        cumulative_sum += pdf[i]
        cdf[i] = cumulative_sum
    return cdf

# Step 5: Calculate Equalized Levels
def calculate_equalized_levels(cdf, gray_levels):
    L = len(gray_levels)
    equalized_levels = np.zeros_like(cdf, dtype=int)
    for i in range(len(cdf)):
        equalized_levels[i] = int(round((L - 1) * cdf[i]))
    return equalized_levels

# Step 6: Apply Equalized Levels
def apply_equalized_levels(image, equalized_levels):
    rows, cols = image.shape
    equalized_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = equalized_levels[image[i, j]]
    return equalized_image

# Step 7: Main Processing
# Calculate histogram
histogram = calculate_histogram(image)

# Total pixels in the image
total_pixels = image.size

# Calculate PDF and CDF
pdf = calculate_pdf(histogram, total_pixels)
cdf = calculate_cdf(pdf)

# Calculate equalized levels
gray_levels = np.arange(256)  # Gray levels from 0 to 255
equalized_levels = calculate_equalized_levels(cdf, gray_levels)

# Apply histogram equalization
equalized_image = apply_equalized_levels(image, equalized_levels)

# Step 8: Visualization
plt.figure(figsize=(12, 6))

# Original image and histogram
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Original Histogram")
plt.bar(range(256), histogram, color='blue', width=1)
plt.xlabel("Gray Level")
plt.ylabel("Frequency")

# Equalized image and histogram
equalized_histogram = calculate_histogram(equalized_image)
plt.subplot(2, 2, 3)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Equalized Histogram")
plt.bar(range(256), equalized_histogram, color='green', width=1)
plt.xlabel("Gray Level")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
