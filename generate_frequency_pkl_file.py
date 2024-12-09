import cv2
import numpy as np
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define the path to the dataset directory
directory = os.path.abspath('dataset')

# List all .jpg files in the directory
jpg_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]

# Function to calculate FFT of an image
def Calculate_FFT(image_path):
    # Read the image in grayscale and resize to fixed size (100x100 pixels)
    target_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(target_image, (100, 100))
    
    # Perform 2D FFT and shift zero frequency component to center
    target_fft = np.fft.fft2(resized_image)
    target_fft_shifted = np.fft.fftshift(target_fft)
    
    # Return the absolute value of the FFT, flattened into a 1D array
    return np.abs(target_fft_shifted).flatten()

# Function to process a single image and return its FFT result
def process_image(image_name):
    image_path = os.path.join(directory, image_name)
    return image_name, Calculate_FFT(image_path)

# Dictionary to store the FFT results for each image
img_fre = {}

# Using ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    # Submit tasks to process images concurrently
    results = list(tqdm(executor.map(process_image, jpg_files), total=len(jpg_files), desc="Processing Images"))

    # Store the results in the dictionary
    for img_name, fft_result in results:
        img_fre[img_name] = fft_result

# Save the FFT results to a pickle file
with open('frequency.pkl', 'wb') as f:
    pickle.dump(img_fre, f)

# Get the file size of the pickle file
file_size = os.path.getsize("frequency.pkl")
file_size_mb = file_size / (1024 * 1024)

# Log the size of the pickle file in MB
print(f"Size of the img_fre_values file: {file_size_mb:.2f} MB")
