from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def test_predict():
    # Crea una imagen de prueba en blanco y negro
    img = Image.new('L', (224, 224), color=255)  # 'L' para escala de grises; color=255 para blanco
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post("/predict", files={"file": ("test.png", img_byte_arr, "image/png")})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_file():
    response = client.post("/predict", files={"file": ("test.txt", b"test content", "text/plain")})
    assert response.status_code == 400