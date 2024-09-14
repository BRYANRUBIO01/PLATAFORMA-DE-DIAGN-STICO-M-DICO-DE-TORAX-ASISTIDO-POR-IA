import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from cnn_model import create_model

def load_data(data_dir, metadata_path, target_shape=(224, 224, 1), test_size=0.2, random_state=42):
    # Cargar metadatos
    metadata = pd.read_csv(metadata_path)
    
    # Remover la extensión '.png' de los nombres en la columna 'Image Index'
    metadata['Image Index'] = metadata['Image Index'].str.replace('.png', '')
    
    # Obtener lista de archivos de imagen disponibles
    available_images = set(f[:-4] for f in os.listdir(data_dir) if f.endswith('.npy'))
    
    # Filtrar metadata para incluir solo las imágenes disponibles
    metadata_filtered = metadata[metadata['Image Index'].isin(available_images)]
    
    print(f"Total de imágenes disponibles: {len(available_images)}")
    print(f"Total de entradas en metadata después del filtrado: {len(metadata_filtered)}")
    
    # Cargar imágenes y etiquetas
    images = []
    labels = []
    skipped_images = 0
    
    for _, row in metadata_filtered.iterrows():
        image_id = row['Image Index']
        filename = f"{image_id}.npy"
        
        if os.path.exists(os.path.join(data_dir, filename)):
            img = np.load(os.path.join(data_dir, filename))
            
            if img.shape != target_shape:
                if img.shape == (1, 224, 224, 1):
                    img = img.squeeze(0)
                elif img.shape == (1, 224, 224, 3):
                    img = img.squeeze(0)
                    img = np.mean(img, axis=-1, keepdims=True)  # Convertir RGB a escala de grises
                else:
                    print(f"Advertencia: La imagen {filename} tiene una forma inconsistente {img.shape}. Saltando esta imagen.")
                    skipped_images += 1
                    continue
            
            images.append(img)
            
            # Obtener etiquetas
            image_labels = row['Finding Labels']
            
            # Convertir etiquetas a formato one-hot
            label_list = image_labels.split('|')
            one_hot_labels = [1 if label in label_list else 0 for label in ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']]
            labels.append(one_hot_labels)
    
    print(f"Imágenes saltadas debido a formas inconsistentes: {skipped_images}")
    
    if not images:
        raise ValueError("No se pudieron cargar imágenes. Verifica las rutas de los archivos y la estructura de los metadatos.")
    
    # Convertir listas a arrays de NumPy
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Imágenes cargadas: {len(images)}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    
    # Dividir en conjuntos de entrenamiento y validación
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return (x_train, y_train), (x_val, y_val)

def train_model(train_data, val_data, epochs=10):
    model = create_model()
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow(
        train_data[0],  # x_train
        train_data[1],  # y_train
        batch_size=32
    )
    
    history = model.fit(
        train_generator,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    data_dir = "data/processed/"
    metadata_path = "data/metadata/Data_Entry_2017.csv"
    
    (x_train, y_train), (x_val, y_val) = load_data(data_dir, metadata_path)
    
    train_data = (x_train, y_train)
    val_data = (x_val, y_val)
    
    model, history = train_model(train_data, val_data)
    model.save("models/chest_xray_model.h5")
    
    print("Modelo entrenado y guardado con éxito.")