# Plataforma de Diagnóstico Médico Asistido por IA

## Descripción
Este proyecto implementa una plataforma de diagnóstico médico utilizando técnicas avanzadas de inteligencia artificial y aprendizaje profundo. La plataforma se centra en el análisis automatizado de radiografías de tórax para detectar condiciones médicas.

## Características
- Modelo de Red Neuronal Convolucional (CNN) para clasificación de radiografías de tórax
- API RESTful para servir predicciones del modelo en tiempo real
- Interfaz de usuario intuitiva para la carga de imágenes y visualización de resultados
- Preprocesamiento de imágenes y aumento de datos para mejorar la robustez del modelo
- Pruebas automatizadas para asegurar la calidad del código
- Visualización local para pruebas rápidas y desarrollo

## Tecnologías Utilizadas
- Python 3.8+
- TensorFlow 2.x
- FastAPI
- React
- scikit-learn
- NumPy
- Pandas
- Pillow
- Matplotlib
- Seaborn
- pytest (para pruebas)

## Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/tu-usuario/medical-diagnosis-ai.git
   cd medical-diagnosis-ai
   ```

2. Crear y activar un entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Descargar el conjunto de datos:
   El conjunto de datos de radiografías de tórax de NIH se puede obtener de [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). Descarga y coloca los datos en la carpeta `data/raw/`.

## Uso

1. Preprocesar los datos:
   ```
   python src/data/preprocess.py
   ```

2. Entrenar el modelo:
   ```
   python src/models/train_model.py
   ```

3. Iniciar la API:
   ```
   uvicorn app.main:app --reload
   ```

4. Visualización local:
   Abre el archivo `app/static/index.html` en tu navegador para interactuar con la interfaz de usuario localmente.

## Ejecución de pruebas

Para ejecutar las pruebas automatizadas:

```
pytest tests/
```

Esto ejecutará todas las pruebas en el directorio `tests/`, incluyendo pruebas unitarias para el modelo y pruebas de integración para la API.

## Estructura del Proyecto
```
medical-diagnosis-ai/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── visualization/
├── app/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Contribuir
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos antes de hacer un pull request.

## Licencia
Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto
[Tu Nombre] - [tu.email@ejemplo.com]

Enlace del proyecto: [https://github.com/tu-usuario/medical-diagnosis-ai](https://github.com/tu-usuario/medical-diagnosis-ai)

## Agradecimientos
- [NIH Clinical Center](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) por proporcionar el conjunto de datos de radiografías de tórax.
- Todos los contribuyentes que han participado en este proyecto.
