# Frequency Matching Image Search

A computer vision application that performs frequency-based image similarity search using Fast Fourier Transform (FFT) and cosine similarity. Built with Python, OpenCV, NumPy, and scikit-learn with a Tkinter GUI.

---

## Overview

This system allows users to upload an image and instantly find the most visually similar images from a dataset — not by pixel comparison, but by comparing their **frequency domain representations** using FFT. This approach is robust to minor color and lighting variations, focusing instead on structural and textural similarity.

---

## How It Works

```
User uploads image
        │
        ▼
Convert to grayscale + resize to 100x100
        │
        ▼
Apply Fast Fourier Transform (FFT)
        │
        ▼
Compute frequency magnitude spectrum
        │
        ▼
Compare via cosine similarity against pre-computed dataset FFTs
        │
        ▼
Return top-N most similar images
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Image Processing | OpenCV, NumPy |
| Frequency Analysis | NumPy FFT (np.fft.fft2) |
| Similarity Metric | Cosine Similarity (scikit-learn) |
| GUI | Tkinter |
| Performance | Parallel processing (multiprocessing) |
| Dataset | Describing Textures Dataset (DTD) |

---

## Features

- **FFT-based matching** — compares images in the frequency domain for structural similarity
- **Cosine similarity** — robust similarity metric unaffected by magnitude scaling
- **Parallel processing** — multiprocessing speeds up FFT computation across large datasets
- **Tkinter GUI** — simple drag-and-drop interface for image upload and result display
- **Pre-computed cache** — FFT data stored in pickle file for fast repeated queries

---

## Project Structure

```
Frequency-Matching/
├── main.py                        # Tkinter GUI — image upload + similarity search
├── generate_frequency_pkl_file.py # Pre-computes FFTs for dataset, saves to pickle
├── arrange_images.py              # Organizes DTD dataset into working directory
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- DTD Dataset ([download here](https://www.robots.ox.ac.uk/~vgg/data/dtd/))

### 1. Clone the repository

```bash
git clone https://github.com/battu2001/Frequency-Matching.git
cd Frequency-Matching
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and organize the dataset

Download the DTD dataset and unzip into the project directory, then run:

```bash
python arrange_images.py
```

### 4. Pre-compute FFT data

```bash
python generate_frequency_pkl_file.py
```

This generates `frequency.pkl` — a cached file of FFT representations for all dataset images.

### 5. Run the application

```bash
python main.py
```

---

## Usage

1. Launch the app — a Tkinter window opens
2. Click **Upload Image** and select a `.jpg` or `.jpeg` file
3. The app computes the FFT of your image and searches the dataset
4. Top matching images are displayed ranked by cosine similarity score

---

## Technical Details

### FFT-based Image Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    fft = np.fft.fft2(resized)
    magnitude = np.abs(fft).flatten()
    return magnitude

# Cosine similarity between query and dataset
similarity = cosine_similarity([query_fft], dataset_ffts)
```

### Why FFT for Image Matching?

- Pixel-based comparison is sensitive to minor shifts, rotations, and lighting
- FFT captures **frequency content** (edges, textures, patterns) which is more stable
- Cosine similarity normalizes for magnitude differences — only the pattern matters

---

## Performance

| Metric | Value |
|---|---|
| Image resize | 100x100 px |
| FFT computation | ~2ms per image |
| Dataset search | Parallel (multiprocessing) |
| Input format | .jpg / .jpeg |

---

## Contributors

- Meghana Battu
- Venkata Sai Rohith Pagadala

*Built as part of the M.S. Computer Science program at Florida Atlantic University*

---
