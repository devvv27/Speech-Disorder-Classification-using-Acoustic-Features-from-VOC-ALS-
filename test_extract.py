import os
from app import extract_acoustic_features
wav = r'c:\DewangData\ProjectsTHISSEM\Speech Processing\Speech Project\uploads\CT001_phonationO.wav'
print('Testing WAV:', wav)
res = extract_acoustic_features(wav)
print('Result:', res)
