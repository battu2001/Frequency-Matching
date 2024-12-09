import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import pickle
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import threading
from tqdm import tqdm

with open('frequency.pkl', 'rb') as f:
    img_fre = pickle.load(f)

images_dir = os.path.abspath('dataset')

root = tk.Tk()
root.title("Frequency Matching")

desc_label = tk.Label(root, text="Upload an image and get similar frequency matches!", font=("Arial", 14))
desc_label.pack(pady=10)

slider_label = tk.Label(root, text="Slider Value: 5", font=("Arial", 12))
slider_label.pack(pady=10)

slider = tk.Scale(root, from_=1, to=10, orient="horizontal", length=300)
slider.set(5)
slider.pack(pady=10)

# Variable to store the path of the uploaded image
uploaded_image_path = None

# Label to display "Please wait..." text between Submit and Refresh button
wait_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
wait_label.pack(pady=10)

def clear_previous():
    global uploaded_image_path
    uploaded_image_path = None  # Reset the uploaded image path
    label.config(image='', text="")
    for widget in root.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.destroy()
    slider.set(5)  # Reset slider to 5
    slider_label.config(text="Slider Value: 5")
    wait_label.config(text="")

def upload_image():
    global uploaded_image_path  # Access the global variable
    clear_previous() 
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg")])
    if file_path:
        uploaded_image_path = file_path  # Store the path of the uploaded image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        label.config(text="Uploaded Image", font=("Arial", 12))
        slider_value = slider.get()
        slider_label.config(text=f"Slider Value: {slider_value}")

def Calculate_FFT(image_path):
    target_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(target_image, (100, 100))
    target_fft = np.fft.fft2(resized_image)
    target_fft_shifted = np.fft.fftshift(target_fft)
    return np.abs(target_fft_shifted).flatten().reshape(1, -1)

def display_similar_images(uploaded_image_path, n):
    if not os.path.exists(images_dir):
        print("Directory 'image' not found!")
        return

    input_fre = Calculate_FFT(uploaded_image_path)
    cs = []
    image_names = []
    for i in img_fre:
        cs.append(cosine_similarity(img_fre[i].reshape(1, -1), input_fre.reshape(1, -1))[0][0])
        image_names.append(i)

    arr_cs = np.array(cs)
    arr_image = np.array(image_names)

    sorted_indices = arr_cs.argsort()[::-1]
    top_n_images = arr_image[sorted_indices[:n]]

    # Clear the "Please wait..." label
    wait_label.config(text="")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    for i, img_name in enumerate(top_n_images):
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        img = img.resize((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(frame, image=img_tk, text=f"Similar {i+1}", compound="top")
        img_label.image = img_tk
        img_label.grid(row=0, column=i, padx=10)

def submit_action():
    global uploaded_image_path
    if uploaded_image_path is None:
        # Show an error message if no image was uploaded
        messagebox.showerror("Error", "No image uploaded! Please upload an image first.")
    else:
        img = Image.open(uploaded_image_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        label.config(text="Uploaded Image", font=("Arial", 12))

        slider_value = slider.get()
        slider_label.config(text=f"Slider Value: {slider_value}")
        
        # Show loading indicator
        wait_label.config(text="Please wait... Loading images...")
        
        # Use threading to avoid blocking the UI
        threading.Thread(target=process_and_display, args=(uploaded_image_path, slider_value)).start()

def process_and_display(uploaded_image_path, slider_value):
    # This function runs the heavy computation
    display_similar_images(uploaded_image_path, slider_value)
    
    # After the computation is complete, we can leave the images in the UI
    # The "Please wait..." label has already been cleared in display_similar_images

# Refresh action
def refresh_action():
    clear_previous()  # Reset everything
    wait_label.config(text="")  # Clear the wait message

# Buttons for upload, submit, and refresh
upload_button = tk.Button(root, text="Upload JPG Image", command=upload_image)
upload_button.pack(pady=10)

# Label to display the uploaded image (between slider and upload button)
label = tk.Label(root)
label.pack(pady=10)

submit_button = tk.Button(root, text="Submit", command=submit_action)
submit_button.pack(pady=10)

# Add a refresh button
refresh_button = tk.Button(root, text="Refresh", command=refresh_action)
refresh_button.pack(pady=10)

root.mainloop()
