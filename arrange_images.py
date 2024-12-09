import os
import shutil

# Define the parent directory where the images are currently located
parent_dir = os.path.abspath('dtd/images')

# Define the new directory where images will be moved to
new_dir = os.path.abspath('dataset')

# Ensure the new directory exists, create it if it doesn't
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def move_images_to_new_dir(parent_dir, new_dir):
    print(f"Starting the image moving process from {parent_dir} to {new_dir}")
    
    # Walk through the directory tree from top to bottom
    for root, dirs, files in os.walk(parent_dir, topdown=True):
        # Print the current directory and files being processed
        print(f"Checking directory: {root}")
        print(f"Files found: {files}")
        
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff')):
                file_path = os.path.join(root, file)  # Get the full path of the file
                destination_path = os.path.join(new_dir, file)  # Destination path in new dir
                
                # Check if a file with the same name already exists in the new directory
                if not os.path.exists(destination_path):
                    shutil.move(file_path, destination_path)  # Move the file
                    print(f'Moved: {file_path} to {new_dir}')
                else:
                    print(f'Skipped (duplicate found): {file_path}')
            else:
                print(f"Skipping non-image file: {file}")
    
    print(f"Image moving process completed from {parent_dir} to {new_dir}")

# Call the function to move images
move_images_to_new_dir(parent_dir, new_dir)
