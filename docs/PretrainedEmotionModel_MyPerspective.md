# How I Applied a Pre‑Trained Emotion Recognition Model (Keras/TensorFlow)

## Why I used a pre-trained model
I wanted fast, reliable emotion recognition without spending hours training from scratch. My goal was to use an existing model (a saved Keras `.h5` checkpoint) and plug it into my pipeline for quick image and real-time webcam predictions.

## My goals
- Load a pre-trained model that predicts the 7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise.
- Keep preprocessing consistent with how the model was trained (48×48 grayscale in my case).
- Run inference on both image files and a webcam feed.
- Evaluate on my validation set to verify performance.

## My setup (Windows)
I work on Windows with Python and TensorFlow. I used a virtual environment and installed the core packages I need:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install "tensorflow>=2.9,<2.17" opencv-python numpy scipy scikit-learn
```

## The model I used
I reused my best checkpoint from training, saved as `emotion_model_best.h5` (fallback: `emotion_model_final.h5`). That’s effectively a pre-trained model for my setup. You can also drop in any compatible `.h5` model trained for FER (as long as inputs/labels match).

- Labels I use (order matters):
  - `["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]`
- Input format: 48×48 grayscale, pixel values scaled to [0, 1]

## Loading the pre-trained model (minimal example)
```python
from tensorflow.keras.models import load_model

model = load_model("emotion_model_best.h5")  # or "emotion_model_final.h5"
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

## My preprocessing and inference pipeline (image file)
I detect faces with OpenCV, crop, convert to grayscale, resize to 48×48, normalize, and then run `model.predict`.

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model_best.h5")
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img = cv2.imread("path_to_image.jpg")
if img is None:
    raise RuntimeError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape (1, 48, 48, 1)

    probs = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(labels[idx], float(probs[idx]))
```

## Real-time webcam inference (what I run)
In my project, `main.py` already has a “Real-time detection” option. When I select it, it loads the pre-trained `.h5` model and applies the same preprocessing to each face it detects from the webcam stream.

Behind the scenes, it does the same steps as the image example: detect → crop → grayscale 48×48 → normalize → predict → overlay label and confidence on the frame.

## How I evaluate the model
I use the menu option “Evaluate model on validation set” in `main.py`. I set the validation generator to `shuffle=False` so the predictions line up with labels. The evaluation prints:
- Overall accuracy
- Per-class accuracy (to see which emotions are weak/strong)
- Optional classification report and confusion matrix (if scikit-learn is available)

This helps me spot biases (e.g., over-predicting happy/neutral) and decide whether to continue training.

## When I needed more accuracy
If validation results weren’t good enough, I used “Continue training from saved model” in `main.py`. My training uses class weighting and label smoothing to reduce bias toward dominant classes. I keep the learning rate small and use early stopping/checkpoints so I don’t overfit.

## Tips I follow
- Keep the preprocessing exactly the same between training and inference (48×48 grayscale and scaling to [0, 1]).
- Ensure label order matches the model’s output order.
- Check face detection—if the crop is off, predictions suffer. Better detectors (e.g., MediaPipe/RetinaFace) can help.
- If I switch to a transfer-learning backbone (e.g., MobileNetV2/EfficientNet), I change the input to RGB 224×224 and use the corresponding `preprocess_input`.

## Optional: quick transfer-learning head (for later)
If I need a stronger backbone, I can build a MobileNetV2 head like this (then fine-tune on my data):

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.2)(x)
out = layers.Dense(7, activation="softmax")(x)
model = models.Model(inputs=base.input, outputs=out)
```

This model won’t be good at emotions until I fine-tune it with my dataset, but it gives me a strong starting point.

## What worked for me
- Using my best saved checkpoint (`emotion_model_best.h5`) to skip long initial training.
- Running the built-in evaluation to see per-class accuracy and confusion.
- Continuing training with class weights + label smoothing to reduce happy/neutral bias.
- Keeping preprocessing consistent and validating with `shuffle=False`.

## What I’ll try next
- Add TensorBoard/CSV logging to track training curves.
- Experiment with a MobileNet/EfficientNet backbone and fine-tune for 10–30 epochs.
- Try a stronger face detector and mild alignment to improve crops.
- Consider TensorFlow Lite for speed if I need lower latency.
