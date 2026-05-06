import pandas as pd
import os

try:
    a = pd.read_csv('voc_als_acoustic_features.csv')
    m = pd.read_csv('VOC-ALS/metadata.csv')
except Exception as e:
    print('ERROR reading files:', e)
    raise

print('features shape:', a.shape)
print('features columns:', len(a.columns))
print('first 12 columns:', list(a.columns[:12]))

na_counts = a.isna().sum().sort_values()
print('\nTop 10 columns with fewest NaNs:')
print(na_counts.head(10))
print('\nTop 10 columns with most NaNs:')
print(na_counts.tail(10))

# prepare metadata file_id
m['file_id'] = m['file_path'].apply(lambda x: os.path.basename(x).replace('.wav','') if pd.notna(x) else None)
merged = pd.merge(a, m[['file_id','meanF0Hz','Jitter','Shimmer','HNR','Label']], on='file_id', how='left')
print('\nmerged shape:', merged.shape)

if 'meanF0Hz' in merged.columns:
    diff_f0 = (merged['f0_mean'] - merged['meanF0Hz']).abs()
    print('mean abs diff f0_mean vs meanF0Hz (non-NaN):', diff_f0.dropna().mean())
    print('median abs diff:', diff_f0.dropna().median())
    print('rows missing metadata meanF0Hz:', merged['meanF0Hz'].isna().sum())

# label agreement
if 'Label' in merged.columns:
    agree = (merged['label'] == merged['Label']).mean()
    print('label agreement fraction (label vs metadata Label):', agree)

print('\nSample comparison (first 10 merged rows):')
print(merged[['file_id','f0_mean','meanF0Hz','label','Label']].head(10).to_string(index=False))

print('\nDone verification.')
