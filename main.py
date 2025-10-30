import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_data_dir = 'Data/train'
validation_data_dir = 'Data/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=30,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Create train generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),  # Standard size for emotion recognition
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

# Create validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Define class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Compute class weights to handle class imbalance
def compute_class_weights(generator):
    """Return dict class_index -> weight computed from generator.classes"""
    classes = generator.classes
    class_counts = np.bincount(classes)
    num_classes = len(class_counts)
    total = classes.shape[0]
    # Avoid division by zero in case a class has zero samples (shouldn't happen if folders exist)
    weights = {i: (total / (num_classes * count)) for i, count in enumerate(class_counts) if count > 0}
    print("Class counts:", {k: int(v) for k, v in enumerate(class_counts)})
    print("Class weights:", {k: round(v, 3) for k, v in weights.items()})
    return weights

# Print generator information
print("Class indices:", train_generator.class_indices)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)
print("Number of classes:", train_generator.num_classes)

# Check if trained model already exists
def build_and_train_model():
    """Build and train an improved CNN model"""
    print("🔧 Building improved emotion detection model...")
    
    # Build an improved CNN model (fixed architecture)
    model = Sequential()

    # First Convolutional Block - Increased filters
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Block - More filters
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Block - Even more filters
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Flatten and Dense layers - Improved architecture
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotion classes

    # Compile with better optimizer settings
    from tensorflow.keras.optimizers import Adam
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Improved training callbacks
    checkpoint = ModelCheckpoint('emotion_model_best.h5', 
                                monitor='val_accuracy', 
                                save_best_only=True, 
                                mode='max', 
                                verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                  patience=15,  # More patience
                                  restore_best_weights=True,
                                  verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.5,  # Less aggressive reduction
                                 patience=7, 
                                 min_lr=0.00001,
                                 verbose=1)

    # Train the model with more epochs and class weighting (to handle imbalance)
    print("🚀 Starting improved training (this will take longer but give better results)...")
    print("📊 Target: >60% accuracy for good emotion detection")
    class_weights = compute_class_weights(train_generator)
    history = model.fit(
        train_generator,
        epochs=100,  
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # Save the final model
    model.save('emotion_model_final.h5')
    print("✅ Training completed!")
    print("📈 Check the final validation accuracy above")
    return model


def continue_training_from_saved(epochs: int = 20, initial_lr: float = 1e-4):
    """
    Continue training from an already saved model (prefer the best checkpoint).
    Args:
        epochs: Additional epochs to train.
        initial_lr: Learning rate to use when resuming.
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    # Decide which model file to load
    model_path = None
    if os.path.exists('emotion_model_best.h5'):
        model_path = 'emotion_model_best.h5'
    elif os.path.exists('emotion_model_final.h5'):
        model_path = 'emotion_model_final.h5'
    elif os.path.exists('emotion_model.h5'):
        model_path = 'emotion_model.h5'

    if model_path is None:
        print("❌ No existing model found to continue training. Please train from scratch first.")
        return None

    print(f"🔁 Loading model from '{model_path}' to continue training...")
    model = load_model(model_path)

    # Re-compile with a conservative LR for fine-tuning
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Callbacks: keep saving the best onto the same best file
    checkpoint = ModelCheckpoint('emotion_model_best.h5',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-6,
                                  verbose=1)

    print(f"🚀 Continuing training for {epochs} more epochs (LR={initial_lr})...")
    class_weights = compute_class_weights(train_generator)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # Optionally update the final model snapshot as well
    model.save('emotion_model_final.h5')
    print("✅ Continued training completed and model snapshot saved to 'emotion_model_final.h5'")
    return model

# Check if model already exists
model_exists = (
    os.path.exists('emotion_model_best.h5') or
    os.path.exists('emotion_model_final.h5') or 
    os.path.exists('emotion_model.h5')
)

if model_exists:
    print("✅ Trained model found! Skipping training...")
    print("📊 Dataset Info:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {validation_generator.samples}")
    print(f"   Classes: {list(train_generator.class_indices.keys())}")
else:
    print("❌ No trained model found. Training required...")
    

# Function for real-time emotion detection
def detect_emotion_realtime():
    """
    Real-time facial emotion detection using webcam
    """
    # Load the trained model - try best models first
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('emotion_model_best.h5'):
            model = load_model('emotion_model_best.h5')
            print("✅ Best model loaded successfully!")
        elif os.path.exists('emotion_model_final.h5'):
            model = load_model('emotion_model_final.h5')
            print("✅ Final model loaded successfully!")
        elif os.path.exists('emotion_model.h5'):
            model = load_model('emotion_model.h5')
            print("✅ Model loaded successfully!")
        else:
            print("❌ No model found. Please train first (option 1).")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 for model input
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)
            
            # Predict emotion
            prediction = model.predict(face_input, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion_label = class_labels[emotion_index]
            confidence = prediction[0][emotion_index] * 100
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display emotion label and confidence
            label_text = f"{emotion_label}: {confidence:.1f}%"
            cv2.putText(frame, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Facial Emotion Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Function to detect emotion from image file
def detect_emotion_from_image(image_path):
    """
    Detect emotion from a static image file
    """
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('emotion_model_best.h5'):
            model = load_model('emotion_model_best.h5')
            print("✅ Best model loaded successfully!")
        elif os.path.exists('emotion_model_final.h5'):
            model = load_model('emotion_model_final.h5')
            print("✅ Final model loaded successfully!")
        elif os.path.exists('emotion_model.h5'):
            model = load_model('emotion_model.h5')
            print("✅ Model loaded successfully!")
        else:
            print("❌ No model found. Please train first (option 1).")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    for (x, y, w, h) in faces:
        # Extract and process face
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        
        # Predict emotion
        prediction = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_label = class_labels[emotion_index]
        confidence = prediction[0][emotion_index] * 100
        
        # Draw results on image
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label_text = f"{emotion_label}: {confidence:.1f}%"
        cv2.putText(image, label_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display result
    cv2.imshow('Emotion Detection Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate_on_validation():
    """Evaluate the best available model on the validation set and print metrics."""
    try:
        from tensorflow.keras.models import load_model
    except Exception as e:
        print(f"❌ Cannot import Keras load_model: {e}")
        return

    # Load model (prefer best)
    model_path = None
    if os.path.exists('emotion_model_best.h5'):
        model_path = 'emotion_model_best.h5'
    elif os.path.exists('emotion_model_final.h5'):
        model_path = 'emotion_model_final.h5'
    elif os.path.exists('emotion_model.h5'):
        model_path = 'emotion_model.h5'
    else:
        print("❌ No model found to evaluate. Train or continue training first.")
        return

    print(f"🔍 Evaluating model: {model_path}")
    model = load_model(model_path)

    # Predict on validation set (ensure no shuffle on validation generator)
    steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
    preds = model.predict(validation_generator, steps=steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = validation_generator.classes[:len(y_pred)]

    # Overall accuracy
    overall_acc = (y_pred == y_true).mean()
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    print("Per-class accuracy:")
    for idx, name in enumerate(class_labels):
        mask = (y_true == idx)
        if mask.sum() == 0:
            print(f" - {name}: no samples")
        else:
            acc_i = (y_pred[mask] == idx).mean()
            print(f" - {name}: {acc_i:.4f}")

    # Confusion matrix / report (if sklearn available)
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
    except Exception:
        print("sklearn not installed; skipping detailed report.")

# Main execution
if __name__ == "__main__":
    print("🎭 Facial Emotion Detection Program")
    print("="*50)
    
    # Check if we need to train or can proceed to menu
    if not model_exists:
        print("🔄 No trained model found. You must train first before using detection features.")
    
    print("\n🚀 Choose an option:")
    if model_exists:
        print("1. 🔁 Continue training from saved model (recommended)")
        print("2. 🔄 Retrain from scratch (time-consuming)")
        print("3. 📹 Real-time emotion detection (webcam)")
        print("4. 📸 Detect emotion from image file")
        print("5. 📊 Evaluate model on validation set")
        print("6. ❌ Exit")
    else:
        print("1. 🔄 Train model (required first time)")
        print("2. ❌ Exit")
    
    if model_exists:
        choice = input("\nEnter your choice (1/2/3/4/5/6): ").strip()
    else:
        choice = input("\nEnter your choice (1/2): ").strip()
    
    if choice == "1" and model_exists:
        # Continue training
        try:
            more_epochs = input("Enter extra epochs to train (default 20): ").strip()
            more_epochs = int(more_epochs) if more_epochs else 20
        except Exception:
            more_epochs = 20
        print(f"🔁 Continuing training for {more_epochs} more epochs...")
        continue_training_from_saved(epochs=more_epochs, initial_lr=1e-4)
        print("✅ Continue training completed!")

    elif choice == "2" and model_exists:
        print("🔄 Retraining from scratch... This will take a while.")
        build_and_train_model()
        print("✅ Training completed!")
        
    elif choice == "3" and model_exists:
        print("📹 Starting real-time emotion detection...")
        detect_emotion_realtime()
        
    elif choice == "4" and model_exists:
        image_path = input("📁 Enter path to image file: ").strip()
        detect_emotion_from_image(image_path)
        
    elif choice == "5" and model_exists:
        print("📊 Evaluating on validation set...")
        evaluate_on_validation()
        
    elif choice == "6" and model_exists:
        print("👋 Goodbye!")
        
    elif choice == "1" and not model_exists:
        print("Starting model training... This may take a while.")
        build_and_train_model()
        print("Training completed!")

    elif choice == "2" and not model_exists:
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the program again.")


