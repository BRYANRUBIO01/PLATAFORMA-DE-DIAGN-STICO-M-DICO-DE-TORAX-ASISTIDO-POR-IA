from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Permite solicitudes desde tu aplicación React
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Carga el modelo
model = load_model("models/chest_xray_model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Convertir la imagen a escala de grises
    image = image.convert("L")  # 'L' para escala de grises
    image = image.resize((224, 224))
    
    # Convertir la imagen a un arreglo NumPy
    image_array = np.array(image) / 255.0  # Normalizar la imagen
    image_array = np.expand_dims(image_array, axis=-1)  # Añadir un canal
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión del batch
    
    predictions = model.predict(image_array)
    
    # Mapear las predicciones a las clases correspondientes
    classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 
               'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
               'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
    
    results = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)