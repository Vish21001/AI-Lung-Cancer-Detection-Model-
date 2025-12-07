import os
import cv2
import numpy as np
import pandas as pd

DATA_DIR = "data/"
IMG_SIZE = 128  # Resize CT scans
ANNOTATIONS_FILE = "annotations.csv"  # CSV with nodule labels

def load_images():
    """
    Load images and labels from LIDC-IDRI dataset
    Assumes annotations.csv has columns: filename, label (0=healthy, 1=cancer)
    """
    df = pd.read_csv(ANNOTATIONS_FILE)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(row['label'])
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
    labels = np.array(labels)
    return images, labels
