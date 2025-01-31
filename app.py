import os
import shutil
import time
import random
import datetime

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from skimage.measure import label, regionprops, find_contours
from skimage.util import random_noise
import cloudinary
import cloudinary.uploader
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch.nn as nn
import pickle

# Local imports (ensure these files exist in your project)
from utils import *
from network import deeplabv3plus_resnet101

###############################################################################
# APP CONFIGURATION
###############################################################################
app = Flask("__name__", static_folder="build", static_url_path="/")

# Folder configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["CORS_HEADERS"] = "Content-Type, auth"
app.config["CORS_RESOURCES"] = {r"*": {"origins": "*"}}
app.config["CORS_METHODS"] = "GET,POST,OPTIONS"
app.config["CORS_SUPPORTS_CREDENTIALS"] = True

CORS(app, resources={r"*": {"origins": "*"}})

# Cloudinary configuration
cloudinary.config(
    cloud_name="dviezxheb",
    api_key="562469498594536",
    api_secret="hu1bLkjakFmolb9UURsgjuyhDJo"
)

# Allowed file extensions (currently unused in the code)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

###############################################################################
# GLOBAL VARIABLES / MAPPINGS
###############################################################################
# Labels used for a specific classification route
all_labels = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
key_to_label_mapper = {i: label for i, label in enumerate(all_labels)}

# Set random seeds for reproducibility
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def _build_cors_prelight_response():
    """
    Builds a response for CORS preflight requests, allowing all origins,
    headers, and methods.
    """
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def read_image(image_path):
    """
    Reads an image from the file path, resizes it, and applies basic
    pre-processing (median-based contrast shift + normalization).
    Returns a 4D NumPy array suitable for Keras/TensorFlow models.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 384))
    # Simple contrast shift by median
    image = np.clip(image - np.median(image) + 127, 0, 255)
    # Normalize to [0, 1]
    image = image / 255.0
    image = image.astype(np.float32)
    # Expand dimensions to match (batch, height, width, channels)
    image = np.expand_dims(image, axis=0)
    return image


def parse(y_pred):
    """
    Expands the dimensions of the predicted mask so that it can be
    visualized or processed further.
    """
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1].astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def get_colored_segmentation_image(image, pred):
    """
    Draws contours around the predicted areas and annotates the detection.
    `pred` is assumed to be a single-channel mask in [0, 255].
    """
    _, binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Put label text
    cv2.putText(image, "Fire Detection", (10, 360), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    # Draw contours
    cv2.drawContours(image, contours, -1, (0, 0, 255), 5)
    return image


def to_3d(mask):
    """
    Converts a binary 2D mask to a 3D mask by stacking the mask into RGB channels.
    """
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def mask_to_border(mask_3d):
    """
    Converts a 3-channel binary mask to its border by finding contours.
    Returns a 2D (single-channel) border mask.
    """
    # Use only the first channel
    mask = mask_3d[:, :, 0]
    h, w = mask.shape
    border = np.zeros((h, w), dtype=np.uint8)

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border


def mask_to_bbox(mask_3d):
    """
    Finds bounding boxes around the connected regions of a 3-channel mask.
    """
    bboxes = []
    border_mask = mask_to_border(mask_3d)
    lbl = label(border_mask)
    props = regionprops(lbl)
    for prop in props:
        x1, y1, x2, y2 = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])
    return bboxes


 
class CustomModule(nn.Module): 
    def __init__(self, modelName, in_features, out_features, pretrained=True):
        super().__init__()
        self.model = self.chooseModel(modelName, in_features, out_features, pretrained)

    def chooseModel(self, modelName, in_features, out_features, pretrained): 
        if modelName == 'resnet': 
            model = models.resnet18(pretrained=pretrained)
            if in_features > 3: 
                model.conv1 = nn.Conv2d(in_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = model.fc.in_features 
            model.fc = nn.Linear(in_features=num_ftrs, out_features=out_features) 
        else: 
            print('No model given!')
        return model

    def forward(self, x):
        outputs = self.model(x) 
        return outputs 




def predict_image_resnet(image_path):
    """
    Loads a PyTorch ResNet model and predicts the class of the given image.
    Returns the predicted class index and the associated probability (in %).
    """
    print("Prediction in progress...")

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    transformation = torch.nn.Sequential(
        torch.nn.Identity()  # Replace or add transforms if needed
    )

    image_tensor = transformation(torch.tensor(np.array(image)).permute(2, 0, 1))
    image_tensor = image_tensor.float().unsqueeze(0)  # (1, 3, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    # Load model
    model2 = torch.load("models/Resnet_model_Torch_Multi.pt", map_location=device)
    model2.eval()

    # Forward pass
    with torch.no_grad():
        output = model2(image_tensor)

    index = output.cpu().numpy().argmax()
    softmax_func = torch.nn.Softmax(dim=1)
    probabilities = softmax_func(output)
    probability = probabilities[0][index].cpu().item() * 100

    print(f"Probability index: {probability:.2f}%")
    return index, probability

###############################################################################
# FLASK ROUTES
###############################################################################
@app.route("/test/", methods=["POST"])
def test():
    """
    Simple test endpoint to verify that the server is running.
    """
    return {"home": "welcome"}


@app.route("/api/v1/detection", methods=["POST", "OPTIONS"])
def detection():
    """
    Endpoint for fire detection (segmentation). Upload an image,
    the model will produce a segmented result, and the code uploads
    the result to Cloudinary. The local file/folder is removed after.
    """
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()

    # Store result info
    cumulative = []
    folder_to_delete = ""
    # Convert uploaded files into a list
    incoming_images = list((request.files).to_dict().values())

    # Create a random folder name suffix
    r_number = random.randint(1000, 9999)
    for each in incoming_images:
        # Save the file locally
        ext = each.filename.rsplit(".", 1)[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        today = datetime.date.today()
        today_str = today.isoformat() + str(r_number)
        new_filename = f"{timestr}.{ext}"
        try:
            os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], today_str))
        except FileExistsError:
            pass

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], today_str, new_filename)
        each.save(file_path)

        folder_to_delete = os.path.join(app.config["UPLOAD_FOLDER"], today_str)

        # Load segmentation model (assumes function is in utils.py)
        model_path = "models/model.h5"
        model = load_model_weight(model_path)

        # Preprocess & predict
        img = read_image(file_path)
        prediction = model.predict(img)[0][..., -1]
        pred = parse(prediction)
        pred = (pred * 255).astype(np.uint8)

        # Draw segmentation
        original_img = cv2.imread(file_path)
        original_img = cv2.resize(original_img, (512, 384))
        pred_color = get_colored_segmentation_image(original_img, pred)
        cv2.imwrite(file_path, pred_color)

        # Upload to Cloudinary
        rez = cloudinary.uploader.upload(file_path)
        cumulative.append({"path": rez["url"]})

    # Clean up local folder
    if folder_to_delete:
        shutil.rmtree(folder_to_delete, ignore_errors=True)

    return {"Prediction": cumulative}


@app.route("/api/v1/classification", methods=["POST", "OPTIONS"])
def create_classification():
    """
    Endpoint for CNN-based classification (example: breast cancer).
    Uses a local Keras model and label binarizer to classify images
    uploaded by the user.
    """
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()

    cumulative = []
    folder_to_delete = ""
    incoming_images = list((request.files).to_dict().values())

    r_number = random.randint(1000, 9999)
    for file in incoming_images:
        # Save file
        ext = file.filename.rsplit(".", 1)[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        today = datetime.date.today()
        today_str = today.isoformat() + str(r_number)
        new_filename = f"{timestr}.{ext}"

        try:
            os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], today_str))
        except FileExistsError:
            pass

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], today_str, new_filename)
        file.save(file_path)
        folder_to_delete = os.path.join(app.config["UPLOAD_FOLDER"], today_str)

        # Load image and Keras model
        image = cv2.imread(file_path)
        output = image.copy()
        image = cv2.resize(image, (128, 128))
        image = img_to_array(image.astype("float"))
        image = np.expand_dims(image, axis=0)

        model = load_model("models/CancerSein_01.h5")
        lb = pickle.loads(open("labelbin", "rb").read())

        # Predict
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label_str = lb.classes_[idx]
        label_info = f"{label_str}: {proba[idx] * 100:.2f}%"

        # Draw result
        output = imutils.resize(output, width=400)
        cv2.putText(output, label_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 3)
        cv2.imwrite(file_path, output)

        # Upload to Cloudinary
        rez = cloudinary.uploader.upload(file_path)
        cumulative.append({"path": rez["url"], "label": label_info})

    if folder_to_delete:
        shutil.rmtree(folder_to_delete, ignore_errors=True)

    return {"Prediction": cumulative}


@app.route("/api/v1/ct_detection", methods=["POST", "OPTIONS"])
def create_ct_detection():
    """
    Endpoint for segmentation (example: CT detection).
    Loads a Deeplabv3+ PyTorch model and marks bounding boxes on the result.
    """
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()

    cumulative = []
    folder_to_delete = ""
    incoming_images = list((request.files).to_dict().values())

    r_number = random.randint(1000, 9999)
    for file in incoming_images:
        # Save file
        ext = file.filename.rsplit(".", 1)[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        today = datetime.date.today()
        today_str = today.isoformat() + str(r_number)
        new_filename = f"{timestr}.{ext}"

        try:
            os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], today_str))
        except FileExistsError:
            pass

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], today_str, new_filename)
        file.save(file_path)
        folder_to_delete = os.path.join(app.config["UPLOAD_FOLDER"], today_str)

        # Example setting seeds and loading model
        #seeding(42)
        num_classes = 3
        checkpoint_path = "models/checkpoint.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = deeplabv3plus_resnet101(num_classes=num_classes, output_stride=16)
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Preprocess
        ct_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        ct_image = cv2.resize(ct_image, (320, 320))
        image_copy = ct_image.copy()

        ct_image = np.transpose(ct_image, (2, 0, 1)) / 255.0
        ct_image = np.expand_dims(ct_image, axis=0).astype(np.float32)
        ct_image_tensor = torch.from_numpy(ct_image).to(device)

        # Inference
        with torch.no_grad():
            pred_y = model(ct_image_tensor)
            pred_y = torch.softmax(pred_y, dim=1)
            pred_y = torch.argmax(pred_y, dim=1)
            pred_y = pred_y[0].cpu().numpy().astype(np.uint8)

        # Mask and bounding boxes
        binary_mask = (pred_y > 0).astype(np.float32)
        binary_mask_3d = to_3d(binary_mask) * 255.0  # Scale to [0,255]
        binary_mask_3d = binary_mask_3d.astype(np.uint8)
        bboxes = mask_to_bbox(binary_mask_3d)

        # Draw bounding boxes
        for bbox in bboxes:
            cv2.rectangle(
                image_copy,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 0, 255),
                3
            )

        # Overlay mask
        alpha = 0.5
        cv2.addWeighted(binary_mask_3d, alpha, image_copy, 1 - alpha, 0, image_copy)

        cv2.imwrite(file_path, image_copy)

        # Upload to Cloudinary
        rez = cloudinary.uploader.upload(file_path)
        cumulative.append({"path": rez["url"]})

    if folder_to_delete:
        shutil.rmtree(folder_to_delete, ignore_errors=True)

    return {"Prediction": cumulative}


@app.route("/api/v1/chest", methods=["POST", "OPTIONS"])
def create_classification_chest():
    """
    Example endpoint for chest X-ray classification using a PyTorch ResNet model.
    """
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()

    cumulative = []
    folder_to_delete = ""
    incoming_images = list((request.files).to_dict().values())

    r_number = random.randint(1000, 9999)
    for file in incoming_images:
        # Save file
        ext = file.filename.rsplit(".", 1)[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        today = datetime.date.today()
        today_str = today.isoformat() + str(r_number)
        new_filename = f"{timestr}.{ext}"

        try:
            os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], today_str))
        except FileExistsError:
            pass

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], today_str, new_filename)
        file.save(file_path)
        folder_to_delete = os.path.join(app.config["UPLOAD_FOLDER"], today_str)

        # Load and classify with PyTorch ResNet
        image = cv2.imread(file_path)
        output = image.copy()

        index_01, proba_idx_data = predict_image_resnet(file_path)
        label_info = f"{key_to_label_mapper[index_01]}: {proba_idx_data:.2f}%"

        output = imutils.resize(output, width=400)
        cv2.putText(output, label_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 3)
        cv2.imwrite(file_path, output)

        # Upload result
        rez = cloudinary.uploader.upload(file_path)
        cumulative.append({"path": rez["url"], "label": label_info})

    if folder_to_delete:
        shutil.rmtree(folder_to_delete, ignore_errors=True)

    return {"Prediction": cumulative}


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    app.run(debug=True)
