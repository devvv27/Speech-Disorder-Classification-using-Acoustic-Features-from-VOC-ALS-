import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating dataset folder with 65-15-15 split")
print()

# ============================================================================
# 1. SCAN ALL SPECTROGRAM IMAGES
# ============================================================================
print("[1/4] Scanning spectrogram images...")

image_paths = []
labels = []

dataset_dir = 'VOC-ALS'
classes = ['ALS', 'HC']
vowels = ['A', 'E', 'I', 'O', 'U']

for class_label in classes:
    for vowel in vowels:
        # Original spectrograms
        spec_dir = os.path.join(dataset_dir, class_label, vowel, 'spectogram')
        if os.path.exists(spec_dir):
            for filename in os.listdir(spec_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(spec_dir, filename))
                    labels.append(class_label)
        
        # Augmented spectrograms
        aug_spec_dir = os.path.join(dataset_dir, class_label, vowel, 'augmented_spectogram')
        if os.path.exists(aug_spec_dir):
            for filename in os.listdir(aug_spec_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(aug_spec_dir, filename))
                    labels.append(class_label)
        
        # Voiced-only spectrograms
        voiced_spec_dir = os.path.join(dataset_dir, class_label, vowel, 'voiced', 'spectograms')
        if os.path.exists(voiced_spec_dir):
            for filename in os.listdir(voiced_spec_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(voiced_spec_dir, filename))
                    labels.append(class_label)

print(f"  Total images found: {len(image_paths)}")
print(f"  Class distribution: ALS={labels.count('ALS')}, HC={labels.count('HC')}")
print()

# ============================================================================
# 2. SPLIT DATA (65-15-15)
# ============================================================================
print("[2/4] Splitting data: 65% train, 15% val, 15% test...")

labels_array = np.array(labels)

# First split: 65% train, 35% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels_array,
    test_size=0.35,
    random_state=42,
    stratify=labels_array
)

# Second split: 50-50 of temp (15% val, 15% test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"  Training: {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
print(f"  Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
print(f"  Test: {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
print(f"  Train - ALS: {np.sum(y_train=='ALS')}, HC: {np.sum(y_train=='HC')}")
print(f"  Val - ALS: {np.sum(y_val=='ALS')}, HC: {np.sum(y_val=='HC')}")
print(f"  Test - ALS: {np.sum(y_test=='ALS')}, HC: {np.sum(y_test=='HC')}")
print()

# ============================================================================
# 3. CREATE FOLDER STRUCTURE & COPY FILES
# ============================================================================
print("[3/4] Creating folder structure and copying files...")

base_output = 'dataset'
splits = {
    'train': (X_train, y_train),
    'val': (X_val, y_val),
    'test': (X_test, y_test)
}

metadata_rows = []

for split_name, (split_paths, split_labels) in splits.items():
    split_dir = os.path.join(base_output, split_name)
    
    for class_label in classes:
        class_dir = os.path.join(split_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)
    
    # Copy files to appropriate class folders
    for src_path, label in tqdm(zip(split_paths, split_labels), total=len(split_paths), desc=f"  Copying {split_name}"):
        # Get filename and check if it's augmented
        filename = os.path.basename(src_path)
        
        # Add "_aug" suffix to augmented spectrograms to avoid overwriting
        if 'augmented_spectogram' in src_path:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_aug{ext}"
        
        # Destination path
        dst_path = os.path.join(split_dir, label, filename)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        
        # Record metadata (relative path for Kaggle)
        rel_path = os.path.relpath(dst_path, base_output)
        metadata_rows.append({
            'image_path': rel_path,
            'label': label,
            'split': split_name
        })

print(f"  ✓ Copied {len(metadata_rows)} files to dataset/")
print()

# ============================================================================
# 4. CREATE METADATA CSV
# ============================================================================
print("[4/4] Creating metadata CSV...")

metadata_df = pd.DataFrame(metadata_rows)
metadata_csv = os.path.join(base_output, 'metadata.csv')
metadata_df.to_csv(metadata_csv, index=False)

print(f"  ✓ Metadata saved: {metadata_csv}")
print(f"  Shape: {metadata_df.shape}")
print(f"  Split distribution:\n{metadata_df['split'].value_counts()}")
print(f"  Label distribution:\n{metadata_df['label'].value_counts()}")
print()

print("Dataset structure:")
for root, dirs, files in os.walk(base_output):
    level = root.replace(base_output, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in sorted(files)[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')

print()
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset creation complete!")
print(f"Ready to upload to Kaggle: {base_output}/")
