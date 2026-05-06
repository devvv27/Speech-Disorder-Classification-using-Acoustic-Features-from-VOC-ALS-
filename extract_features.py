import parselmouth
import os
import numpy as np
import pandas as pd
from parselmouth.praat import call
from sklearn.decomposition import PCA
from numpy import linalg
from python_speech_features import delta
from scipy.stats import describe
from itertools import product
import glob
import re

window_size = 0.025
time_step = 0.01

def compute_complexity_array(win, m1, m2):
    arr_0 = np.zeros((win, win), np.float64)
    m1 = (m1 - m1.mean()) / (m1.std() * len(m1))
    m2 = (m2 - m2.mean()) / (m2.std())
    for n1 in range(win):
        for n2 in range(win):
            num = 0.0
            for i in range(len(m1)):
                num += (m1[i-n1])*(m2[i-n2])
            arr_0[n1, n2] = num
    return arr_0

def get_f0(pitch):
    arr = pitch.selected_array["frequency"]
    return pd.DataFrame({
        "f0_mean": call(pitch, "Get mean", 0, 0, "Hertz"),
        "f0_median": np.median(arr[arr != 0]),
        "f0_sd": call(pitch, "Get standard deviation", 0, 0, "Hertz"),
        "f0_cv": call(pitch, "Get standard deviation", 0, 0, "Hertz") / call(pitch, "Get mean", 0, 0, "Hertz"),
        "f0_slope": pitch.get_mean_absolute_slope(),
        "f0_p5": call(pitch, "Get quantile", 0, 0, 0.05, "Hertz"),
        "f0_p95": call(pitch, "Get quantile", 0, 0, 0.95, "Hertz"),
        "f0_p5-95": float(call(pitch, "Get quantile", 0, 0, 0.95, "Hertz")) - float(call(pitch, "Get quantile", 0, 0, 0.05, "Hertz")),
    }, index=[0])

def get_formants(voiced, pitch, formant_range=5):
    formants = voiced.to_formant_burg(window_length=window_size, time_step=time_step)
    formant_values = []
    formant_dict = {}
    for i in range(1, formant_range + 1):
        formant_value = np.asarray([formants.get_value_at_time(formant_number=i, time=t) for t in pitch.xs()])
        formant_values.append(formant_value)
        formant_dict[f"f{i}_mean"] = np.nanmean(formant_value)
        formant_dict[f"f{i}_median"] = np.nanmedian(formant_value)
        formant_dict[f"f{i}_std"] = np.nanstd(formant_value)
        formant_dict[f"f{i}_prc_5"] = np.nanquantile(formant_value, q=0.05)
        formant_dict[f"f{i}_prc_95"] = np.nanquantile(formant_value, q=0.95)
        formant_dict[f"f{i}_prc_5_95"] = np.nanquantile(formant_value, q=0.95) - np.nanquantile(formant_value, q=0.05)
    result = pd.DataFrame(formant_dict, index=[0])
    return result, formant_values

def get_jitter(pointProcess):
    return pd.DataFrame({
        "rapJitter": call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) * 100,
        "localJitter": call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100,
        "localabsoluteJitter": call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) * 100,
        "ppq5Jitter": call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) * 100,
        "ddpJitter": call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) * 100,
    }, index=[0])

def get_shimmer(voiced, pointProcess):
    return pd.DataFrame({
        "localShimmer": call([voiced, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100,
        "localdbShimmer": call([voiced, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "apq3Shimmer": call([voiced, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100,
        "aqpq5Shimmer": call([voiced, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100,
        "apq11Shimmer": call([voiced, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100,
        "ddaShimmer": call([voiced, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100,
    }, index=[0])

def get_harmonicity(voiced):
    harmonicity = call(voiced, "To Harmonicity (cc)", 0.01, 75, 0.1, 1)
    return pd.DataFrame({
        "hnr_mean": call(harmonicity, "Get mean", 0, 0),
        "hnr_sd": call(harmonicity, "Get standard deviation", 0, 0),
    }, index=[0])

def get_formant_slopes(formant_values, times, num_formants=3):
    formant_slopes = []
    formant_slopes_dict = {}
    for i in range(num_formants):
        formant_slope = np.gradient(formant_values[i], times * 20)
        formant_slopes.append(formant_slope)
        formant_num = i + 1
        formant_slopes_dict[f"f{formant_num}_d_dx_median"] = np.nanmedian(formant_slope)
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_5"] = np.nanquantile(formant_slope, q=0.05)
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_95"] = np.nanquantile(formant_slope, q=0.95)
        formant_slopes_dict[f"f{formant_num}_d_dx_prc_5_95"] = np.nanquantile(formant_slope, q=0.95) - np.nanquantile(formant_slope, q=0.05)
    result = pd.DataFrame(formant_slopes_dict, index=[0])
    return result, formant_slopes

def get_formant_accels(formant_slope_values, times, num_formants=3):
    formant_accels_dict = {}
    for i in range(num_formants):
        formant_accel = np.gradient(formant_slope_values[i], times * 20)
        formant_num = i + 1
        formant_accels_dict[f"f{formant_num}_dd_dx_median"] = np.nanmedian(formant_accel)
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_5"] = np.nanquantile(formant_accel, q=0.05)
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_95"] = np.nanquantile(formant_accel, q=0.95)
        formant_accels_dict[f"f{formant_num}_dd_dx_prc_5_95"] = np.nanquantile(formant_accel, q=0.95) - np.nanquantile(formant_accel, q=0.05)
    return pd.DataFrame(formant_accels_dict, index=[0])

def get_intensity(voiced):
    intensity = call(voiced, "To Intensity...", 100, 0.0)
    return pd.DataFrame({
        "intensity_mean_dB": call(intensity, "Get mean...", 0.0, 0.0, "dB"),
        "intensity_sd_dB": call(intensity, "Get standard deviation...", 0.0, 0.0),
        "intensity_cv_dB": call(intensity, "Get standard deviation...", 0.0, 0.0) / call(intensity, "Get mean...", 0.0, 0.0, "dB"),
    }, index=[0])

def get_MFCCs(sound):
    n_mfcc = 1 + 13
    mfcc_object = sound.to_mfcc(number_of_coefficients=n_mfcc - 1)
    mfcc0 = mfcc_object.to_array().T
    mfccs = np.c_[mfcc0, delta(mfcc0, 2)]
    tsc_mfcc = np.sum(np.gradient(mfcc0[:, 1:], axis=0) ** 2, axis=1)
    tsc_mfcc_df = pd.DataFrame({"tsc_mfcc_mean": tsc_mfcc.mean(), "tsc_mfcc_cv": tsc_mfcc.std() / tsc_mfcc.mean()}, index=[0])
    _, _, mu, sig, _, _ = describe(mfccs, axis=0)
    mfcc_stats = np.r_[mu, sig]
    mfcc_summary_df = pd.DataFrame(data=mfcc_stats.T, index=["_".join([x, y]) for x, y in list(product(["mean", "var"], ["mfcc_" + str(k) for k in range(n_mfcc)] + ["mfcc_slope_" + str(k) for k in range(n_mfcc)]))]).T
    return pd.concat([tsc_mfcc_df, mfcc_summary_df], axis=1)

def interp_formant(formant_values):
    if any(np.isnan(formant_values)):
        formant_values = pd.Series(formant_values).interpolate(method="linear", limit_direction="both").values
    return formant_values

def nearest(short_arr, long_arr, tol=0.001):
    match_vals = []
    for s in short_arr:
        new_val = np.argmin(np.abs(s - long_arr))
        if np.min(np.abs(s - long_arr)) > tol:
            print("INTENSITY: Detected delta >1ms")
        match_vals.append(new_val)
    return np.asarray(match_vals).ravel()

def get_complexity(arrays, plot=False):
    win = 15
    for ix_a, a in enumerate(arrays):
        if np.sum(np.isnan(a)) > 0:
            a1 = pd.Series(a).interpolate("cubic")
            arrays[ix_a] = a1.values
    arr = np.zeros((win * len(arrays), win * len(arrays)), np.float64)
    for ixm1, m1 in enumerate(arrays):
        for ixm2, m2 in enumerate(arrays):
            arr_0 = compute_complexity_array(win, m1, m2)
            arr[(ixm1 * win):((ixm1 + 1) * win), (ixm2 * win):((ixm2 + 1) * win)] = arr_0
    pca = PCA()
    pca.fit_transform(arr)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    complexity_idx = np.where(cum_var > 0.95)[0][0]
    return complexity_idx

def get_complexity_measures(arrays, array_pairs):
    complexity_measures = {}
    for pair in array_pairs:
        first = pair[0]
        second = pair[1]
        try:
            complexity_measures[f"{first}_{second}_comp"] = get_complexity([arrays[first], arrays[second]])
        except KeyError:
            pass
    return pd.DataFrame(complexity_measures, index=[0])

def extract_acoustic_features(wav_file, f0_min=75, f0_max=600):
    try:
        file = wav_file.replace("\\", "/")
        data_dir = os.path.dirname(file)
        f_old = os.path.basename(file)
        fnew = f_old.replace(".wav", "_OnlyVoiced.wav")
        
        sound = parselmouth.Sound(f"{data_dir}/{f_old}")
        voiced_file = f"{data_dir}/voiced/{fnew}"
        
        if not os.path.exists(voiced_file):
            return None
        
        voiced = parselmouth.Sound(voiced_file)
        
        pitch = call(voiced, "To Pitch (cc)", 0.02, f0_min, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f0_max)
        pitch_values = pitch.selected_array["frequency"]
        pitch_times = np.asarray(pitch.xs())
        
        pointProcess = call([voiced, pitch], "To PointProcess (cc)")
        
        jitter_df = get_jitter(pointProcess)
        shimmer_df = get_shimmer(voiced, pointProcess)
        harmonicity_df = get_harmonicity(voiced)
        f0_df = get_f0(pitch)
        formants_df, formant_values = get_formants(voiced, pitch)
        formant_slopes_df, formant_slope_values = get_formant_slopes(formant_values, pitch_times)
        formant_accels_df = get_formant_accels(formant_slope_values, pitch_times)
        intensity_df = get_intensity(voiced)
        mfccs_df = get_MFCCs(sound)
        
        voiced_norm = voiced.copy()
        call(voiced_norm, "Scale intensity...", 70.0)
        intensity_norm = call(voiced_norm, "To Intensity...", 100, 0.001)
        match_vals = nearest(short_arr=pitch_times, long_arr=intensity_norm.xs())
        intensities_final = intensity_norm.values[0][match_vals]
        
        comp_arrays = {
            "F0": pitch_values,
            "F1": interp_formant(formant_values[0]),
            "F2": interp_formant(formant_values[1]),
            "F3": interp_formant(formant_values[2]),
            "IN": intensities_final,
        }
        
        comp_array_pairs = [("F0", "F1"), ("F0", "F2"), ("F0", "F3"), ("F1", "F2"), ("F1", "F3"), ("F2", "F3"), ("F0", "IN")]
        complexities_df = get_complexity_measures(comp_arrays, comp_array_pairs)
        
        return pd.concat([jitter_df, shimmer_df, harmonicity_df, f0_df, formants_df, formant_slopes_df, formant_accels_df, intensity_df, mfccs_df, complexities_df], axis=1)
    except Exception as e:
        print(f"Error processing {wav_file}: {str(e)}")
        return None

def main():
    voc_als_dir = "VOC-ALS"
    metadata_path = os.path.join(voc_als_dir, "metadata.csv")
    
    metadata = pd.read_csv(metadata_path)
    
    all_features = []
    all_labels = []
    all_file_ids = []
    all_subjects = []
    all_ages = []
    all_sexes = []
    all_severities = []
    
    for cls in ["ALS", "HC"]:
        cls_dir = os.path.join(voc_als_dir, cls)
        for vowel in ["A", "E", "I", "O", "U"]:
            vowel_dir = os.path.join(cls_dir, vowel)
            if not os.path.exists(vowel_dir):
                continue
            wav_files = glob.glob(os.path.join(vowel_dir, "*.wav"))
            for wav_file in wav_files:
                if "_OnlyVoiced" in wav_file:
                    continue
                print(f"Processing {wav_file}...")
                features = extract_acoustic_features(wav_file)
                if features is not None:
                    file_id = os.path.basename(wav_file).replace(".wav", "")
                    subject_match = re.search(r'(PZ\d+|CT\d+)', file_id)
                    subject_id = subject_match.group(1) if subject_match else None
                    
                    meta_row = metadata[metadata['subjectID'] == subject_id]
                    if len(meta_row) > 0:
                        age = meta_row.iloc[0]['Age']
                        sex = meta_row.iloc[0]['Sex']
                        severity = meta_row.iloc[0].get('Severity', np.nan)
                    else:
                        age = np.nan
                        sex = np.nan
                        severity = np.nan
                    
                    all_features.append(features)
                    all_labels.append(cls)
                    all_file_ids.append(file_id)
                    all_subjects.append(subject_id)
                    all_ages.append(age)
                    all_sexes.append(sex)
                    all_severities.append(severity)
    
    if all_features:
        features_df = pd.concat(all_features, ignore_index=True)
        features_df["label"] = all_labels
        features_df["file_id"] = all_file_ids
        features_df["subject_id"] = all_subjects
        features_df["Age"] = all_ages
        features_df["Sex"] = all_sexes
        features_df["Severity"] = all_severities
        features_df.to_csv("voc_als_acoustic_features.csv", index=False)
        print(f"Saved {len(features_df)} feature rows to voc_als_acoustic_features.csv")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
