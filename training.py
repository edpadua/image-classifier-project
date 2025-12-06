import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configurações ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Exemplo: Gato e Cachorro
MODEL_PATH = 'image_classifier_model.h5'

def build_and_train_model():
    # 1. Carregar o Modelo Base (VGG16) sem a camada superior
    # weights='imagenet' - usa pesos pré-treinados
    # include_top=False - exclui a última camada de classificação
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # 2. Congelar as camadas do modelo base (Transfer Learning)
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Adicionar novas camadas de classificação (Custom Head)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # 4. Criar o modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compilar o modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # --- SIMULAÇÃO DE CARREGAMENTO DE DADOS (Substitua pela sua lógica real) ---
    # Aqui, você usaria ImageDataGenerator para carregar seus dados de treinamento e validação.
    # Exemplo (assumindo que você tem uma pasta 'data/train' e 'data/validation'):
    if not os.path.exists('data/train'):
        print("Caminho 'data/train' não encontrado. Usando dados de teste simulados.")
        # Se você não tiver dados reais, não execute o treinamento, apenas o salvamento (como abaixo).
        # Para um projeto real, você DEVE ter dados aqui.
        return model 

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # 6. Treinar o modelo (simulação)
    # Você faria um `model.fit` aqui:
    # model.fit(train_generator, epochs=5)
    
    # 7. Salvar o modelo e o mapeamento de classes
    model.save(MODEL_PATH)
    class_indices = train_generator.class_indices
    # Salve as classes para que a API saiba o que é classe 0, 1, etc.
    with open('class_map.txt', 'w') as f:
        f.write(str(class_indices))
    
    print(f"Modelo e classes salvos em: {MODEL_PATH}")
    return model

if __name__ == '__main__':
    # Execute este script para treinar e salvar o modelo antes de rodar a API
    build_and_train_model()