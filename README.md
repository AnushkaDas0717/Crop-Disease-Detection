DataSet: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# Plant Disease Detection Project

## Overview
The **Plant Disease Detection Project** uses machine learning and deep learning techniques to automatically detect diseases in plants. This project helps farmers and agricultural researchers identify plant diseases at an early stage, enabling effective intervention and improving crop yield.

## Features
- Detects a variety of common plant diseases from images of leaves.
- Provides a user-friendly interface for disease diagnosis.
- Based on a pre-trained deep learning model.
- Designed for use in both research and real-world farming scenarios.

## Technologies Used
- **Python**: Programming language for developing the backend.
- **TensorFlow**: Deep learning framework for model training and prediction.
- **Keras**: High-level neural networks API used with TensorFlow.
- **OpenCV**: Computer vision library for image processing and augmentation.
- **NumPy, Pandas**: Data handling and manipulation.
- **Matplotlib, Seaborn**: Data visualization and model performance evaluation.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv myenv
myenv\Scripts\activate  # For Windows
source myenv/bin/activate  # For macOS/Linux
```

### Step 3: Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

## Usage
1. **Input an image** of a plant leaf to the model.
2. The model will process the image and provide a **disease prediction**.
3. The prediction result will display the disease name and associated details (if available).


## Model Information
The model used in this project is a Convolutional Neural Network (CNN) trained on a dataset of plant leaf images. The training process involves:
1. Image augmentation techniques to improve model generalization.
2. Transfer learning using pre-trained models  **ResNet**.
3. Fine-tuning the model to predict various plant diseases.

## Contributions
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

### To Contribute:
1. Fork the repository
2. Create a new branch
3. Make changes and commit them
4. Submit a pull request


## Acknowledgments
- **TensorFlow**, **Keras**, and **OpenCV** for providing the libraries to build and train models.
- **new-plant-diseases-dataset**
