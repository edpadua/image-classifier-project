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
cd image-classifier-project
2. Configure Virtual EnvironmentIt is highly recommended to use a virtual environment (venv) to manage dependencies:Bash# 1. Create the virtual environment
python -m venv venv

# 2. Activate the virtual environment (Windows)
.\venv\Scripts\activate

# 2. Activate the virtual environment (Linux/macOS)
# source venv/bin/activate
3. Install DependenciesCreate a requirements.txt file with the following content and install the libraries:Plaintexttensorflow==2.16.1
keras==3.3.3
numpy
pillow
fastapi
uvicorn
python-multipart
h5py
Bashpip install -r requirements.txt
4. Prepare the DataCreate the data structure for training. The training.py script expects subfolders to act as classes:Bashmkdir -p data/train/cats
mkdir -p data/train/dogs
# ... place your categorized images inside these folders.
5. Train and Save the ModelRun this script to generate the model (image_classifier_model.h5) and the class mapping (class_map.txt). This step is mandatory before running the API.Bashpython training.py
6. Start the APIStart the web server using uvicorn:Bashuvicorn app:app --reload
The API will be available at http://127.0.0.1:8000.🌐 How to Use the API1. Web Interface (Quick Test)Access http://127.0.0.1:8000/ to use the simple HTML form to upload and test the image classification.2. Interactive Documentation (Swagger UI)Access http://127.0.0.1:8000/docs to see the interactive FastAPI documentation and test the /predict/ endpoint directly.Main EndpointMethodPathDescriptionPOST/predict/Receives an image (multipart/form-data) and returns the predicted class and confidence score.Example Response (JSON):JSON{
  "filename": "my_cat.jpg",
  "prediction": "cats",
  "confidence": "98.50%",
  "all_probabilities": {
    "dogs": 0.015,
    "cats": 0.985
  }
}
🧑‍💻 ContributionsContributions are welcome! Feel free to open issues or submit pull requests for improvements such as:Adding Celery for asynchronous processing of large images.Implementing Caching with Redis for repeated inference results.Supporting other Transfer Learning architectures (e.g., ResNet50, EfficientNet).
