import os
import shutil
import uuid
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from scipy.stats import describe
from sklearn.preprocessing import StandardScaler
from python_speech_features import delta as speech_delta
import warnings
from extract_features import extract_acoustic_features as reference_extract_acoustic_features
import importlib.util
import sys

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = joblib.load('logistic_regression_model.pkl')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

scaler = None
feature_columns = None

try:
    training_path = os.path.join(os.path.dirname(__file__), 'voc_als_acoustic_features_mapped.csv')
    training_df = pd.read_csv(training_path)
    drop_cols = ["label", "file_id", "subject_id", "Sex", "Age", "Severity", "Severity_imputed"]
    training_x = training_df.drop(columns=[c for c in drop_cols if c in training_df.columns])
    training_x = training_x.select_dtypes(include=[np.number])
    training_x = training_x.fillna(training_x.mean())
    training_x = training_x.replace([np.inf, -np.inf], np.nan)
    training_x = training_x.fillna(training_x.mean())
    scaler = StandardScaler()
    scaler.fit(training_x)
    feature_columns = list(training_x.columns)
    print(f"✓ Loaded training scaler with {len(feature_columns)} features")
except Exception as e:
    print(f"🛑 Could not load training scaler: {e}")

# Feature extraction function
def extract_acoustic_features(wav_file_path, debug=False):
    try:
        fallback_used = False
        temp_root = os.path.join(app.config['UPLOAD_FOLDER'], f"tmp_{uuid.uuid4().hex}")
        voiced_dir = os.path.join(temp_root, "voiced")
        os.makedirs(voiced_dir, exist_ok=True)

        original_name = secure_filename(os.path.basename(wav_file_path))
        copied_wav = os.path.join(temp_root, original_name)
        shutil.copyfile(wav_file_path, copied_wav)

        # Locate the local Praat scripts copied into this project
        base_dir = os.path.abspath(os.path.dirname(__file__))
        praat_dir = os.path.join(base_dir, "feature_extraction_helpers")
        praat_dir = os.path.normpath(praat_dir)
        praat_script = os.path.join(praat_dir, "extract_voiced_segments.praat")
        textgrid_script = os.path.join(praat_dir, "textgrid2py.praat")

        # Try running the Praat textgrid helper and voiced-extraction script directly
        start_time = 0.1
        f_old = os.path.basename(copied_wav)
        try:
            # First get an estimated start_time via textgrid script
            tg_path = temp_root.replace('\\', '/') + '/' + f_old
            print(f"Running textgrid script: {textgrid_script} on {tg_path}")
            print("Exists copied wav?", os.path.exists(copied_wav), copied_wav)
            print("Textgrid script exists?", os.path.exists(textgrid_script), textgrid_script)
            tg_out = parselmouth.praat.run_file(textgrid_script, tg_path, 50, 0.0, -25.0, 0.01, 0.01, "silent", "sounding", return_variables=True)
            try:
                start_time = tg_out[1].get('start', start_time)
            except Exception:
                start_time = 0.1
        except Exception as e:
            print(f"textgrid helper failed: {e}")
            # debug: list temp_root and praat_dir contents
            try:
                print("temp_root listing:", os.listdir(temp_root))
            except Exception:
                pass

        try:
            # Call the voiced-extraction Praat script: args -> main_dir (directory), sound (filename), silenceStart, silenceEnd
            print(f"Running voiced extraction script: {praat_script} main_dir={temp_root} filename={f_old} start_time={start_time}")
            print("Praat script exists?", os.path.exists(praat_script), praat_script)
            parselmouth.praat.run_file(praat_script, temp_root.replace('\\','/'), f_old, "0", str(start_time), return_variables=True)
        except Exception as e:
            print(f"voiced extraction helper failed: {e}")
            # debug listing
            try:
                print("voiced dir listing before fallback:", os.listdir(voiced_dir))
            except Exception:
                pass
            # Fallback: copy the original into the voiced folder as a best-effort _OnlyVoiced.wav
            try:
                voiced_copy_name = original_name.replace(".wav", "_OnlyVoiced.wav")
                voiced_copy_path = os.path.join(voiced_dir, voiced_copy_name)
                shutil.copyfile(copied_wav, voiced_copy_path)
                print(f"Fallback: copied voiced file to {voiced_copy_path}")
                fallback_used = True
            except Exception as e2:
                print(f"Fallback copy failed: {e2}")

        # Now call the canonical extractor (it will look for temp_root/voiced/<name>_OnlyVoiced.wav)
        features_df = reference_extract_acoustic_features(copied_wav)
        if features_df is None:
            shutil.rmtree(temp_root, ignore_errors=True)
            return None

        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        # Save a CSV copy of the extracted features for parity debugging
        try:
            csv_path = os.path.join(temp_root, "extracted_features_app.csv")
            features_df.to_csv(csv_path, index=False)
        except Exception:
            csv_path = None
        result = features_df.iloc[0].to_dict()
        meta = { 'fallback_used': bool(fallback_used), 'csv_path': csv_path }
        if debug:
            # Return result, the full dataframe as dict for debugging, and meta
            df_dict = features_df.to_dict(orient='records')[0]
            # do not remove temp if debugging so user can inspect files
            return result, df_dict, meta
        # cleanup
        shutil.rmtree(temp_root, ignore_errors=True)
        return result, meta
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            if 'temp_root' in locals() and os.path.exists(temp_root):
                shutil.rmtree(temp_root, ignore_errors=True)
        except:
            pass
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files are supported'}), 400
    
    try:
        if scaler is None or feature_columns is None:
            return jsonify({'error': 'Training scaler could not be loaded'}), 500

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        debug_flag = False
        if request.form.get('debug') in ['1', 'true', 'True']:
            debug_flag = True

        features_out = extract_acoustic_features(filepath, debug=debug_flag)
        if features_out is None:
            return jsonify({'error': 'Failed to extract features from audio file'}), 400

        # unpack returned values (now includes meta info about fallback)
        if debug_flag:
            features, raw_features, meta = features_out
        else:
            features, meta = features_out
        
        # Create a DataFrame from extracted features
        features_df = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)

        # Standardize with the same training-time scaler
        X_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        class_probabilities = dict(zip(model.classes_, probability))
        confidence = class_probabilities.get(str(prediction), max(probability)) * 100
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        response = {
            'prediction': str(prediction),
            'probability': {
                'Disease': float(class_probabilities.get('Disease', 0.0)),
                'No Disease': float(class_probabilities.get('No Disease', 0.0))
            },
            'confidence': float(confidence)
        }
        # Include fallback indicator so clients can know if Praat failed and we used the fallback copy
        response['fallback_used'] = bool(meta.get('fallback_used', False))
        if debug_flag:
            response['extracted_features'] = raw_features
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.remove(filepath)
        except:
            pass
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
