import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

# Butterworth Low Pass Filter
def butterworth_low_pass_filter(shape, D0, n):
    M, N = shape
    H = np.zeros((M, N))
    center_x, center_y = M // 2, N // 2
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            H[u, v] = 1 / (1 + (D / D0)**(2 * n))
    return H

# Gaussian Low Pass Filter
def gaussian_low_pass_filter(shape, D0):
    M, N = shape
    H = np.zeros((M, N))
    center_x, center_y = M // 2, N // 2
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            H[u, v] = np.exp(-D**2 / (2 * D0**2))
    return H

# Apply filter in the frequency domain
def apply_filter(image, filter_kernel):
    # Perform FFT
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # Apply the filter in the frequency domain
    filtered_dft = dft_shift * filter_kernel
    
    # Perform Inverse FFT
    idft_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.abs(np.fft.ifft2(idft_shift))
    
    return filtered_image

# Main code
# Load grayscale image
image = cv2.imread('image.tif', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Add Gaussian noise
noisy_image = add_gaussian_noise(image)

# Create Butterworth and Gaussian filters
D0 = 50  # Cutoff frequency
n = 4     # Butterworth filter order
butterworth_filter = butterworth_low_pass_filter(noisy_image.shape, D0, n)
gaussian_filter = gaussian_low_pass_filter(noisy_image.shape, D0)

# Apply filters
butterworth_filtered_image = apply_filter(noisy_image, butterworth_filter)
gaussian_filtered_image = apply_filter(noisy_image, gaussian_filter)

# Display results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(2, 2, 2), plt.imshow(noisy_image, cmap='gray'), plt.title("Noisy Image")
plt.subplot(2, 2, 3), plt.imshow(butterworth_filtered_image, cmap='gray'), plt.title("Butterworth Filtered")
plt.subplot(2, 2, 4), plt.imshow(gaussian_filtered_image, cmap='gray'), plt.title("Gaussian Filtered")
plt.show()
