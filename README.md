# 🖼️ Image Classifier with Transfer Learning (FastAPI & TensorFlow)

## 🌟 Project Overview

This project implements an **Image Classifier** utilizing the **Transfer Learning** technique (specifically the VGG16 architecture) and served through a fast and efficient **RESTful API**, built with **FastAPI**.

The goal is to classify new images into trained categories (e.g., Dogs and Cats) after applying fine-tuning to a model pre-trained on a massive dataset.

## 🚀 Key Features

* **Transfer Learning:** Uses pre-trained VGG16 weights (from ImageNet) to accelerate training and boost accuracy on a custom dataset.
* **Fast Inference API:** The `/predict/` endpoint uses **FastAPI** to handle image uploads and return predictions as JSON responses.
* **Decoupled Service:** Clear separation between the **Training** logic (`training.py`) and the **API Service** (`app.py`), reflecting MLOps best practices.

---

## 📁 Repository Structure

The project is organized as follows:
/image-classifier-project ├── /data/ # Training dataset (e.g., /train/cats, /train/dogs) ├── /models/ # ML artifacts │ └── image_classifier_model.h5 ├── app.py # Main API code (FastAPI) ├── training.py # Script to train and save the model ├── requirements.txt # Python dependencies ├── class_map.txt # Mapping of indices to class names └── README.md # This file

---

## 🛠️ Setup and Installation Guide

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)

2. Configure Virtual Environment
It is highly recommended to use a virtual environment (venv) to manage dependencies:

# 1. Create the virtual environment
python -m venv venv

# 2. Activate the virtual environment (Windows)
.\venv\Scripts\activate

# 2. Activate the virtual environment (Linux/macOS)
# source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file with the necessary libraries and install them:

tensorflow==2.16.1
keras==3.3.3
numpy
pillow
fastapi
uvicorn
python-multipart
h5py

pip install -r requirements.txt

