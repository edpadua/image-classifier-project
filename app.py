from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import io
import json

# --- Configurações da API ---
app = FastAPI(title="Image Classifier API")
MODEL_PATH = 'image_classifier_model.h5'
IMAGE_SIZE = (224, 224)
model = None
class_map = None

# --- Funções de Carregamento e Pré-processamento ---

def load_model_and_classes():
    """Carrega o modelo Keras e o mapeamento de classes."""
    global model, class_map
    try:
        # Carrega o modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso.")

        # Carrega o mapeamento de classes
        with open('class_map.txt', 'r') as f:
            data = f.read()
            # Converte a string de volta para dicionário
            class_map_raw = json.loads(data.replace("'", "\"")) 
            # Inverte o mapeamento para ter {indice: nome_da_classe}
            class_map = {v: k for k, v in class_map_raw.items()}
            print(f"Classes carregadas: {class_map}")

    except Exception as e:
        print(f"Erro ao carregar modelo ou classes: {e}")
        # Se o modelo não foi encontrado, provavelmente o script training.py não foi executado
        raise RuntimeError("O modelo não foi encontrado. Execute 'python training.py' primeiro.")

def preprocess_image(img_file: bytes):
    """Abre o arquivo, redimensiona e pré-processa para o modelo."""
    try:
        # Abre a imagem do byte stream
        img = Image.open(io.BytesIO(img_file)).convert('RGB')
        # Redimensiona para o tamanho esperado pelo modelo
        img = img.resize(IMAGE_SIZE)
        # Converte para array numpy
        img_array = keras_image.img_to_array(img)
        # Adiciona dimensão de batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        # Pré-processamento (normalização de 0-255 para 0-1) - essencial para VGG16/Transfer Learning
        img_array /= 255.0  

        return img_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {e}")

# --- Endpoints da API ---

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo ao iniciar a API."""
    load_model_and_classes()

@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Página inicial simples com formulário de upload."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classificador de Imagens</title>
    </head>
    <body>
        <h2>Classificador de Imagens Simples com Transfer Learning</h2>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input name="file" type="file" accept="image/*">
            <input type="submit" value="Classificar Imagem">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a previsão."""
    if not model or not class_map:
        raise HTTPException(status_code=503, detail="Modelo não está pronto ou carregado.")

    # 1. Ler o conteúdo do arquivo
    contents = await file.read()
    
    # 2. Pré-processar
    processed_image = preprocess_image(contents)

    # 3. Fazer a previsão
    predictions = model.predict(processed_image)
    
    # 4. Obter a classe com maior probabilidade
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # 5. Mapear o índice para o nome da classe
    predicted_class_name = class_map.get(predicted_class_index, "Classe Desconhecida")
    
    return {
        "filename": file.filename,
        "prediction": predicted_class_name,
        "confidence": f"{confidence * 100:.2f}%",
        "all_probabilities": {class_map.get(i): float(p) for i, p in enumerate(predictions[0])}
    }