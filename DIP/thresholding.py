import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('image.tif', 0)  # Replace 'image.jpg' with your image file
if image is None:
    raise FileNotFoundError("Image not found. Please provide a valid file path.")

# Step 1: Calculate the Histogram
def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.ravel():
        hist[pixel_value] += 1
    return hist

# Step 2: Compute Otsu's Threshold Value
def otsu_threshold(image, hist):
    total_pixels = image.size
    current_max = 0
    threshold = 0
    sum_total = np.sum([i * hist[i] for i in range(256)])
    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if between_variance > current_max:
            current_max = between_variance
            threshold = t
    return threshold

# Step 3: Apply Thresholding
def apply_threshold(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image

# Calculate histogram
histogram = calculate_histogram(image)

# Determine the threshold value
threshold_value = otsu_threshold(image, histogram)
print(f"Calculated Threshold Value: {threshold_value}")

# Perform thresholding
binary_image = apply_threshold(image, threshold_value)

# Step 4: Display Results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Histogram
plt.subplot(1, 3, 2)
plt.title('Histogram')
plt.bar(range(256), histogram, color='black', width=1)
plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Threshold = {threshold_value}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()

# Binary Image
plt.subplot(1, 3, 3)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
