# Frequency Matching Image Search

## Project Title:
**Frequency Matching Image Search**

## Team Members:
- Rohit
- Meghana

## Software Used:
- Python 3.8+
- OpenCV
- NumPy
- Tkinter
- scikit-learn
- Pillow
- tqdm
- concurrent.futures

## Description:
This project allows users to upload an image, and the system calculates the Fast Fourier Transform (FFT) of the image to perform frequency-based image matching. The program finds and displays the most similar images in a dataset based on the cosine similarity of their FFTs.

## Installation Guide:
### Prerequisites:
1. **Install Latest Python Version**:
   - Ensure Python 3.8 or above is installed on your system.
   
2. **Install Dependencies**:
   - Clone the repository and navigate into the project folder.
   - Install the required Python packages using pip by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download the Dataset**:
   - Download the "Describing Textures Dataset (DTD)" from [here](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and unzip it into the project directory.

4. **Organize Images**:
   - Run the script to move the images into the desired directory:
     ```bash
     python arrange_images.py
     ```

5. **Generate Frequency Data**:
   - Run the following script to generate the FFT data for the images and save it into a pickle file:
     ```bash
     python generate_frequency_pkl_file.py
     ```

6. **Run the Application**:
   - To launch the frequency-based image matching application, run the following script:
     ```bash
     python main.py
     ```

## Expected Output:
- When the application is run, a Tkinter window will open where the user can upload an image. The application will process the image's FFT, compare it with the images in the dataset, and display the most similar images based on cosine similarity of their frequency features.

## Input Format:
- **Required Image Format**: `.jpg` or `.jpeg`
- **Image Dimensions**: The images will be resized to 100x100 pixels before processing.

## Operating Systems:
- Windows
- Linux

## Notes:
- The program utilizes parallel processing to speed up the calculation of FFTs for multiple images.
- The GUI is built using Tkinter, allowing users to easily interact with the application.

## References:
- **Dataset**: [Describing Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **FFT Calculation**: Fast Fourier Transform (FFT) explained in detail at [Wikipedia - FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- **Cosine Similarity**: [Cosine Similarity (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## Code Structure:
1. **arrange_images.py**: Organizes the images from the DTD dataset into a new directory.
2. **generate_frequency_pkl_file.py**: Calculates the FFT of each image and stores the result in a pickle file (`frequency.pkl`).
3. **main.py**: A Tkinter-based GUI that allows users to upload images and search for similar images based on FFT frequency matching.

## Comments:
- Reasonable comments have been added throughout the code to help understand the functionality of each function and script.
