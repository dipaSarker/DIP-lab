import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('image.tif', 0)  # Replace 'image.tif' with your file path
if image is None:
    raise FileNotFoundError("Image not found. Please provide a valid file path.")

# Step 1: Calculate the initial threshold manually as the mean of the image intensities
def calculate_initial_threshold(image):
    """
    Calculate the initial threshold value based on the mean intensity of the image.
    Args:
        image: Grayscale image.
    Returns:
        Initial threshold value.
    """
    initial_threshold = np.mean(image)  # Calculate the mean of the image
    return initial_threshold

# Step 2: Basic Global Thresholding Algorithm
def global_threshold(image, initial_threshold=None, max_iterations=100, epsilon=1e-3):
    if initial_threshold is None:
        initial_threshold = calculate_initial_threshold(image)  # Use the mean intensity if no initial threshold is provided

    threshold = initial_threshold
    for _ in range(max_iterations):
        # Separate the image into two regions: above and below the threshold
        below_threshold = image[image <= threshold]
        above_threshold = image[image > threshold]
        
        # Calculate mean intensities for the two regions
        mean_below = below_threshold.mean() if len(below_threshold) > 0 else 0
        mean_above = above_threshold.mean() if len(above_threshold) > 0 else 0
        
        # Compute new threshold as the average of the two means
        new_threshold = (mean_below + mean_above) / 2
        
        # Check for convergence
        if abs(new_threshold - threshold) < epsilon:
            break
        
        threshold = new_threshold
    
    return int(threshold)

# Step 3: Apply Thresholding
def apply_threshold(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image

# Step 4: Execute the Steps
# Find the threshold value using the global thresholding algorithm
threshold_value = global_threshold(image)
print(f"Calculated Threshold Value: {threshold_value}")

# Perform thresholding
binary_image = apply_threshold(image, threshold_value)

# Step 5: Display Results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Thresholded Image
plt.subplot(1, 3, 2)
plt.title(f'Thresholded Image (T = {threshold_value})')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
