import cv2
import os

# Define the absolute path of the image folder
folder = os.path.abspath("../cherry-leaves/healthy")

# Check if the folder exists
if not os.path.exists(folder):
    print(f"Error: The folder '{folder}' does not exist.")
else:
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
