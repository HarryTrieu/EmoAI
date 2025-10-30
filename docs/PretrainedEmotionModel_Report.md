# Using a Pre-trained Emotion Recognition Model (Keras/TensorFlow)

## Executive summary
This report explains how to use a pre-trained facial emotion recognition (FER) model with Keras/TensorFlow in Python, including model options, data preprocessing, inference pipeline, evaluation, and how to integrate a pre-trained model into the existing project. It also covers limitations and next steps to improve accuracy and robustness.

## Objectives
- Load a pre-trained model for 7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise.
- Run inference on webcam frames or image files with consistent preprocessing.
- Evaluate performance on your validation set.
- Avoid re-training from scratch unless desired, reducing time-to-results.

## Prerequisites
- Python 3.8–3.11 (TensorFlow’s Windows support is best in this range)
- Packages: tensorflow>=2.9, opencv-python, numpy, scipy (for some image utilities)
- Optional: scikit-learn (evaluation reports)

Example installation (Windows PowerShell):
```powershell
# Optional: create and activate a venv first
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install "tensorflow>=2.9,<2.17" opencv-python numpy scipy scikit-learn
```

## Model choices
You can use one of these common approaches:

1) Ready-to-use FER model checkpoints (trained on FER2013 or similar)
- mini-XCEPTION (lightweight), VGG-like CNNs, or community .h5 models trained specifically for FER.
- Pros: small, fast; Cons: accuracy varies, sometimes biased toward dominant classes (happy/neutral).

2) Transfer learning from ImageNet backbones
- MobileNetV2, EfficientNet, or ResNet backbones fine-tuned on FER.
- Pros: often higher accuracy and better generalization; Cons: larger input size (e.g., 224×224), heavier models.

3) Your project’s best checkpoint
- If you already trained a model and saved `emotion_model_best.h5` or `emotion_model_final.h5`, that is effectively a “pre-trained” model for your dataset setup. It can be used directly for inference and further fine-tuning.

Recommendation: Start with your best available checkpoint; if performance is insufficient, consider transfer learning using a modern backbone.

## Input format and preprocessing
- Common dataset format: grayscale 48×48 images (FER2013 style). Some transfer learning models expect RGB 224×224.
- Preprocessing for 48×48 grayscale models:
  - Detect face, convert to grayscale
  - Resize to 48×48
  - Scale pixels to [0, 1]
  - Add batch and channel dimensions: shape becomes (1, 48, 48, 1)

- Preprocessing for RGB transfer learning models:
  - Detect face, keep as RGB
  - Resize to backbone input size (e.g., 224×224)
  - Apply model-specific preprocessing (e.g., `tf.keras.applications.*.preprocess_input`)

## Loading a pre-trained model (Keras/TensorFlow)
Below are two minimal examples.

1) Load a local .h5 model (your own checkpoint):
```python
from tensorflow.keras.models import load_model

model = load_model("emotion_model_best.h5")  # or emotion_model_final.h5
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

2) Load a transfer learning model and its head (example: MobileNetV2, 224×224 RGB):
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False  # freeze for fast inference or initial fine-tuning

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.2)(x)
out = layers.Dense(7, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=out)
```
Note: The transfer learning variant above is untrained for FER; you would either load FER-specific weights if available or fine-tune on your data. If you only need “pre-trained on ImageNet” features for immediate testing, expect suboptimal accuracy on emotions until you fine-tune.

## Inference pipeline
1) Detect faces (e.g., OpenCV Haar Cascade or a modern detector like MediaPipe or RetinaFace).
2) Crop/align the face region.
3) Preprocess the crop per the model’s expectation (grayscale 48×48 or RGB 224×224; normalization/preprocess_input).
4) Predict with `model.predict()`.
5) Convert logits to probabilities (softmax) and map to class labels.
6) Render/return the top prediction and confidence.

Minimal grayscale 48×48 example using OpenCV:
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model_best.h5")
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img = cv2.imread("path_to_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)

    probs = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(labels[idx], float(probs[idx]))
```

## Evaluation methodology
- Use a held-out validation set with the same preprocessing used for training.
- Keep validation generator `shuffle=False` to preserve label ordering for metrics.
- Report:
  - Overall accuracy
  - Per-class accuracy (helps spot class imbalance problems)
  - Confusion matrix
  - Optional macro/micro F1-scores
- Consider class weighting and label smoothing during training/fine-tuning to reduce bias toward dominant classes.

## Integration with this project
Your current `main.py` already:
- Loads train/validation generators
- Builds, trains, and saves `emotion_model_best.h5`/`emotion_model_final.h5`
- Supports real-time webcam and single-image inference
- Supports continued training and evaluation with per-class metrics

To use a pre-trained model without retraining:
- Place the `.h5` file in the project root (e.g., `emotion_model_best.h5`).
- Run the program and choose “Real-time detection” or “Detect from image file.”
- If no model is found, you can choose “Continue training” or “Retrain” to fine-tune on your data.

To switch to a transfer-learning backbone:
- Update the data pipeline to RGB 224×224 and replace the current CNN with a MobileNetV2/EfficientNet head.
- Fine-tune for 10–30 epochs with early stopping and a low learning rate (e.g., 1e-4 → 1e-5).

## Known limitations and risks
- Dataset imbalance: FER2013 and similar datasets over-represent “happy/neutral.” Mitigation: class weights, augmentation, focal loss, mixup.
- Domain shift: Webcam lighting, pose, occlusions reduce accuracy compared to curated datasets.
- Expression ambiguity: Subtle or mixed emotions are hard even for humans; expect uncertainty.
- Face detection quality: Missed/partial detections degrade predictions; consider stronger detectors.

## Recommended next steps
- Evaluate your current best model with the built-in “Evaluate on validation set” option.
- If some classes are weak, run “Continue training” with class weights and a small learning rate.
- Consider migrating to a transfer-learning backbone for higher accuracy.
- Add TensorBoard/CSVLogger to track training curves.
- Optionally, export to TensorFlow Lite for faster real-time performance.

## References and further reading
- Keras applications: https://keras.io/api/applications/
- TensorFlow model saving/loading: https://www.tensorflow.org/guide/keras/save_and_serialize
- FER2013 dataset (Kaggle): https://www.kaggle.com/datasets/msambare/fer2013
- OpenCV Haar Cascades: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
- Label smoothing: https://arxiv.org/abs/1512.00567
- Class weighting strategies: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
