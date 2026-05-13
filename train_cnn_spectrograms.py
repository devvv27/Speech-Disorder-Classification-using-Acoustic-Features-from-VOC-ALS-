import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from tqdm import tqdm
import pickle
import argparse
from datetime import datetime

# Arguments
parser = argparse.ArgumentParser(description='Train CNN on spectrograms with 65-15-15 split')
parser.add_argument('--dataset_dir', type=str, default='VOC-ALS', help='Path to VOC-ALS dataset')
parser.add_argument('--epochs', type=int, default=50, help='Maximum epochs (early stopping may stop earlier)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
args = parser.parse_args()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CNN Training Pipeline")
print(f"Dataset: {args.dataset_dir}")
print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}")
print(f"Early Stopping Patience: {args.patience} epochs")
print()

# ============================================================================
# 1. LOAD DATASET - SCAN SPECTROGRAM PNG FILES
# ============================================================================
print("[1/6] Loading spectrograms and labels...")

image_paths = []
labels = []

dataset_path = args.dataset_dir
classes = ['ALS', 'HC']
vowels = ['A', 'E', 'I', 'O', 'U']

for class_label in classes:
    for vowel in vowels:
        # Original spectrograms
        spec_dir = os.path.join(dataset_path, class_label, vowel, 'spectogram')
        if os.path.exists(spec_dir):
            for filename in os.listdir(spec_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(spec_dir, filename))
                    labels.append(class_label)
        
        # Augmented spectrograms
        aug_spec_dir = os.path.join(dataset_path, class_label, vowel, 'augmented_spectogram')
        if os.path.exists(aug_spec_dir):
            for filename in os.listdir(aug_spec_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(aug_spec_dir, filename))
                    labels.append(class_label)

print(f"  Total images found: {len(image_paths)}")
print(f"  Class distribution: ALS={labels.count('ALS')}, HC={labels.count('HC')}")

if len(image_paths) == 0:
    print("ERROR: No spectrogram images found!")
    exit(1)

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
print(f"  Classes: {le.classes_}")
print()

# ============================================================================
# 2. TRAIN-VAL-TEST SPLIT (65-15-15)
# ============================================================================
print("[2/6] Splitting data: 65% train, 15% val, 15% test...")

# First split: 65% train, 35% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, encoded_labels, 
    test_size=0.35, 
    random_state=42, 
    stratify=encoded_labels
)

# Second split: 50-50 of temp (15% val, 15% test from original)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"  Training: {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
print(f"  Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
print(f"  Test: {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
print(f"  Train class dist: ALS={np.sum(y_train==0)}, HC={np.sum(y_train==1)}")
print(f"  Val class dist: ALS={np.sum(y_val==0)}, HC={np.sum(y_val==1)}")
print(f"  Test class dist: ALS={np.sum(y_test==0)}, HC={np.sum(y_test==1)}")
print()

# ============================================================================
# 3. LOAD AND PREPROCESS IMAGES
# ============================================================================
print("[3/6] Loading and preprocessing images...")

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """Load images and preprocess for MobileNetV2"""
    images = []
    valid_indices = []
    
    for i, path in enumerate(tqdm(image_paths)):
        try:
            img = load_img(path, target_size=target_size)
            # Convert grayscale to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
            images.append(img_array)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")
    
    return np.array(images), np.array(valid_indices)

X_train_imgs, train_valid_idx = load_and_preprocess_images(X_train)
X_val_imgs, val_valid_idx = load_and_preprocess_images(X_val)
X_test_imgs, test_valid_idx = load_and_preprocess_images(X_test)

# Keep only valid labels
y_train = y_train[train_valid_idx]
y_val = y_val[val_valid_idx]
y_test = y_test[test_valid_idx]

print(f"  Training images loaded: {X_train_imgs.shape}")
print(f"  Validation images loaded: {X_val_imgs.shape}")
print(f"  Test images loaded: {X_test_imgs.shape}")
print()

# ============================================================================
# 4. BUILD CNN MODEL (MobileNetV2 Transfer Learning)
# ============================================================================
print("[4/6] Building CNN model...")

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=args.learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Model created: {model.count_params()} total parameters")
print(f"  Frozen layers: {sum([1 for layer in base_model.layers if not layer.trainable])} layers")
print()

# ============================================================================
# 5. TRAIN WITH EARLY STOPPING
# ============================================================================
print("[5/6] Training with early stopping...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=args.patience,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_imgs, y_train,
    validation_data=(X_val_imgs, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[early_stop],
    verbose=1
)

print(f"  Training completed after {len(history.history['loss'])} epochs")
print()

# ============================================================================
# 6. EVALUATE ON TEST SET
# ============================================================================
print("[6/6] Evaluating on test set...")

y_pred_prob = model.predict(X_test_imgs)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("[7/7] Saving results...")

# Save model
model.save('cnn_mobilenet_model.h5')
print(f"  ✓ Model saved: cnn_mobilenet_model.h5")

# Save label encoder
with open('cnn_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"  ✓ Label encoder saved: cnn_label_encoder.pkl")

# Results CSV
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Epochs Trained'],
    'Value': [accuracy, precision, recall, f1, len(history.history['loss'])]
})
results_df.to_csv('cnn_mobilenet_results.csv', index=False)
print(f"  ✓ Results saved: cnn_mobilenet_results.csv")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title('CNN Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cnn_mobilenet_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Confusion matrix saved: cnn_mobilenet_confusion_matrix.png")

# Training history plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cnn_mobilenet_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Training history plot saved: cnn_mobilenet_training_history.png")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CNN Training Complete!")
