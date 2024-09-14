import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def preprocess_dataset(raw_dir,processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for fillename in os.listdir(raw_dir):
        image_path=os.path.join(raw_dir, fillename)
        processed_img=preprocess_image(image_path)
        np.save(os.path.join(processed_dir, f"{fillename[:-4]}.npy"),processed_img)

if __name__ == "__main__":
    raw_dir = "data/raw/"
    processed_dir = "data/processed/"
    preprocess_dataset(raw_dir, processed_dir)
