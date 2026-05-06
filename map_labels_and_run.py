import pandas as pd
import os

FEAT='voc_als_acoustic_features.csv'
META='VOC-ALS/metadata.csv'
OUT='voc_als_acoustic_features_mapped.csv'
BACKUP='voc_als_acoustic_features_backup.csv'

if not os.path.exists(FEAT):
    raise SystemExit(f'{FEAT} not found')
if not os.path.exists(META):
    raise SystemExit(f'{META} not found')

print('Loading features...')
fe = pd.read_csv(FEAT)
print('Loading metadata...')
md = pd.read_csv(META)

# ensure file_id exists in features; if not, try to create from file names
if 'file_id' not in fe.columns:
    if 'file_path' in fe.columns:
        fe['file_id'] = fe['file_path'].apply(lambda x: os.path.basename(x).replace('.wav',''))
    else:
        raise SystemExit('file_id column missing in features')

# prepare metadata file_id
if 'file_id' not in md.columns:
    if 'file_path' in md.columns:
        md['file_id'] = md['file_path'].apply(lambda x: os.path.basename(x).replace('.wav',''))
    else:
        raise SystemExit('file_id column missing in metadata')

# merge to get canonical labels
merged = pd.merge(fe, md[['file_id','Label']], on='file_id', how='left', suffixes=('','_meta'))
if merged['Label'].isna().any():
    print('Warning: some rows missing metadata Label; leaving original label for those rows')

# backup original
print('Backing up original features to', BACKUP)
fe.to_csv(BACKUP, index=False)

# replace label column where metadata present
if 'label' in fe.columns:
    fe['label'] = merged['Label'].combine_first(fe['label'])
else:
    fe['label'] = merged['Label']

print('Unique labels after mapping:', fe['label'].unique())
print('Saving mapped features to', OUT)
fe.to_csv(OUT, index=False)

print('Done mapping. Now you can run train_baseline.py using the mapped CSV (it will overwrite if train script uses same name).')
