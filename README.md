# 🖼️ Image Classifier with Transfer Learning (FastAPI & TensorFlow)

![image](https://github.com/edpadua/image-classifier-project/blob/main/image-classifier-capture%20(1).gif)

## 🌟 Project Overview

This project implements an **image classifier** using **Transfer
Learning** with the **VGG16 architecture**, served through a **RESTful
API** built with **FastAPI**.

The goal is to classify new images into trained categories (e.g., *Dogs*
and *Cats*) after applying fine-tuning to a model pre-trained on a large
dataset.

------------------------------------------------------------------------

## 🚀 Key Features

-   **Transfer Learning:** uses pre-trained VGG16 weights (ImageNet) to
    speed up training and improve accuracy.
-   **Fast Inference API:** the `/predict/` endpoint accepts uploaded
    images and returns predictions in JSON format.
-   **Decoupled Architecture:** clear separation between the **training
    logic** (`training.py`) and the **API service** (`app.py`),
    following MLOps best practices.

------------------------------------------------------------------------

## 📁 Repository Structure

    /image-classifier-project
    ├── /data/                     # Training dataset (e.g., /train/cats, /train/dogs)
    ├── /models/                   # ML artifacts
    │   └── image_classifier_model.h5
    ├── app.py                     # Main API code (FastAPI)
    ├── training.py                # Model training and saving script
    ├── requirements.txt           # Python dependencies
    ├── class_map.txt              # Class index-to-name mapping
    └── README.md                  # This file

------------------------------------------------------------------------

## 🛠️ Setup and Installation Guide

### 1. Clone the Repository

``` bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd image-classifier-project
```

------------------------------------------------------------------------

### 2. Set Up the Virtual Environment

``` bash
python -m venv venv
.env\Scriptsctivate
# source venv/bin/activate
```

------------------------------------------------------------------------

### 3. Install Dependencies

    tensorflow==2.16.1
    keras==3.3.3
    numpy
    pillow
    fastapi
    uvicorn
    python-multipart
    h5py

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 4. Prepare the Dataset

``` bash
mkdir -p data/train/cats
mkdir -p data/train/dogs
```

------------------------------------------------------------------------

### 5. Train and Save the Model

``` bash
python training.py
```

------------------------------------------------------------------------

### 6. Start the API

``` bash
uvicorn app:app --reload
```

http://127.0.0.1:8000

------------------------------------------------------------------------

## 🌐 How to Use the API

### 1. Web Interface

http://127.0.0.1:8000/

### 2. Swagger Docs

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## 🔥 Main Endpoint

  Method   Path          Description
  -------- ------------- --------------------------------------------
  POST     `/predict/`   Returns predicted class + confidence score

### Example Response

``` json
{
  "filename": "my_cat.jpg",
  "prediction": "cats",
  "confidence": "98.50%",
  "all_probabilities": {
    "dogs": 0.015,
    "cats": 0.985
  }
}
```

------------------------------------------------------------------------

## 🧑‍💻 Contributions

-   Add Celery for async processing\
-   Add Redis caching\
-   Support more architectures
