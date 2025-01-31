# Dr-annie Backend

## Overview
This backend was developed using **Flask** during the COVID-19 pandemic to support an AI-powered application. The system provides functionalities for disease detection and prediction based on deep learning models. It processes images, performs classifications, and enables secure API-based access to AI-driven diagnosis tools.

## Features
- **AI based Diagnosis**: AI-based classification and detection model.
- **Cloud Storage Integration**: Automatically uploads processed images to Cloudinary.
- **CORS-Enabled API**: Allows secure access across platforms.

## Technologies Used
- **Flask** (Python Web Framework)
- **TensorFlow & Keras** (Deep Learning Models)
- **PyTorch** (ResNet-based Image Classification)
- **OpenCV** (Image Processing)
- **Cloudinary API** (Image Storage)
- **NumPy & Pandas** (Data Processing)
- **Flask-CORS** (Cross-Origin Resource Sharing)

## Installation
To set up the backend, follow these steps:

### Prerequisites
Ensure you have **Python 3.7+** installed. Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

### Running the Application
```sh
python app.py
```

## API Endpoints
The backend exposes the following API endpoints:

### **1. Health Check**
**Endpoint:** `/test/`  
**Method:** `POST`  
**Description:** Checks if the API is running.  


### **2. COVID-19 Classification (X-Ray)**
**Endpoint:** `/api/v1/classification_chest`  
**Method:** `POST`  
**Description:** Classifies X-ray images as COVID-19, Normal, or Pneumonia.

### **3. Breast Cancer Classification**
**Endpoint:** `/api/v1/classification`  
**Method:** `POST`  
**Description:** Detects breast cancer from histopathological images.

### **4. CT Scan Segmentation**
**Endpoint:** `/api/v1/ct_detection`  
**Method:** `POST`  
**Description:** Identifies affected lung regions in CT scans using DeepLabV3+.

## Example Request (Classification)
```sh
curl -X POST "http://localhost:5000/api/v1/classification" -F "image=@path/to/image.jpg"
```
---
## 📞 Contact  
For questions or feedback, feel free to reach out via:  
✉️ **Email:** mengaraaxel@gmail.com 
🔗 **GitHub:** [Author](https://github.com/Gideon-777)  


