# Image Manipulation Detection Backend

This project provides a Flask-based backend for detecting image manipulation using deep learning models. It supports both segmentation (pixel-level localization of manipulated regions) and classification (identifying the type of manipulation).

## Features

- **Segmentation**: Uses DeepLabV3+ with a ResNet50 backbone to generate a binary mask of manipulated areas.
- **Classification**: Uses ResNet50 to classify the image into one of four categories:
  - Splicing
  - Copy-Move
  - Inpainting
  - Face Manipulation
- **API**: Exposes RESTful endpoints for easy integration with frontend applications.
- **Tunneling**: Integrated `ngrok` support to expose the local server publicly.

## Prerequisites

Ensure you have Python 3.11+ installed. The following dependencies are required:

```bash
pip install flask pyngrok torch torchvision torchaudio segmentation-models-pytorch efficientnet-pytorch pillow matplotlib flask-cors
```

## Setup

1.  **Model Weights**:
    - You need to have the pre-trained model weights for both segmentation and classification.
    - Update the paths in `deeplearningmodelrunsetup.ipynb` (or the converted script) to point to your local weight files:
        - `DEEP_LAB_CHECKPOINT`: Path to DeepLabV3+ weights (e.g., `best_deeplabv3_defacto_last.pth`).
        - `CLASSIF_CHECKPOINT`: Path to ResNet50 weights (e.g., `best_resnet50_defacto.pth`).

2.  **Ngrok Token**:
    - If you want to use `ngrok` to expose your server, set your authtoken in the code:
      ```python
      NGROK_TOKEN = "your_ngrok_token_here"
      ```

## Usage

The main entry point is the Jupyter Notebook `deeplearningmodelrunsetup.ipynb`. You can run it directly in a Jupyter environment or convert it to a Python script.

### Running with Jupyter
Open `deeplearningmodelrunsetup.ipynb` and run all cells. The server will start on port 5000 (and via ngrok if configured).

### Running as a Script
If you prefer running it as a standard Python script, export the notebook code to a `.py` file and run:

```bash
python deeplearningmodelrunsetup.py
```

## API Documentation

### 1. Segment Image
**Endpoint**: `/segment`
**Method**: `POST`

**Request**:
- `image`: (File) The image file to analyze.

**Response** (JSON):
- `mask`: Base64 encoded PNG of the binary mask.
- `overlay`: Base64 encoded PNG of the original image with the mask overlaid (green highlight).
- `masked_image`: Base64 encoded PNG of the image with non-manipulated areas blacked out.

### 2. Classify Image
**Endpoint**: `/classify`
**Method**: `POST`

**Request**:
- `image`: (File) The image file to classify.

**Response** (JSON):
- `class_id`: (Integer) The predicted class ID.
- `class_name`: (String) One of "splicing", "copymove", "inpainting", "face".
- `confidence`: (Float) The confidence score of the prediction.

## Project Structure

- `deeplearningmodelrunsetup.ipynb`: Main server code, model loading, and API definitions.
- `trainingFiles/`: Contains notebooks used for training the models (e.g., `Segmentation.ipynb`).
- `model/`: Directory to store model weights (recommended).
