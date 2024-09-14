from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense,Dropout

def create_model(input_shape=(224, 224, 1), num_classes=14):  # Cambiar a 1 canal
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),  # Recibir imágenes de 1 canal
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Uso de sigmoide para problemas multiclase (multietiqueta)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Dado que es un problema de clasificación multietiqueta
                  metrics=['accuracy'])
    
    return model