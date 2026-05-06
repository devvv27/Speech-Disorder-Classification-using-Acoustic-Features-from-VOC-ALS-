# ALS Speech Disorder Classification - Web Application

A professional web application for real-time classification of speech disorders using acoustic features extracted from audio files. The system uses a pre-trained Logistic Regression model to classify audio samples as either "Disease" or "No Disease".

## Features

- **Real-time Audio Processing**: Upload `.wav` files and get instant predictions
- **Acoustic Feature Extraction**: Automatically extracts 115+ acoustic features using Praat (parselmouth)
- **Machine Learning Classification**: Uses pre-trained Logistic Regression model
- **Professional UI**: Clean, intuitive interface with real-time processing feedback
- **Confidence Scores**: Displays prediction confidence and probability distribution
- **Drag-and-Drop Upload**: Easy file upload with drag-and-drop support

## Requirements

- Python 3.8+
- Pre-trained model: `logistic_regression_model.pkl` (must be in the same directory as `app.py`)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_web.txt
```

2. Ensure the following files are in the project root directory:
   - `app.py` (Flask application)
   - `logistic_regression_model.pkl` (trained model)

3. Create the templates directory:
```bash
mkdir templates
```

## Running the Application

1. Navigate to the project directory:
```bash
cd "c:\DewangData\ProjectsTHISSEM\Speech Processing\Speech Project"
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload Audio**: Click the upload area or drag and drop a `.wav` file
2. **Analyze**: Click the "Analyze Audio" button
3. **View Results**: See the classification result with confidence score and probability distribution

## File Structure

```
Speech Project/
├── app.py                          # Flask application
├── requirements_web.txt            # Python dependencies
├── logistic_regression_model.pkl   # Trained model (required)
├── templates/
│   └── index.html                  # Web interface
├── uploads/                        # Temporary upload directory (auto-created)
└── README.md                       # This file
```

## Technical Details

### Feature Extraction
The application extracts the following acoustic features:

- **Fundamental Frequency (F0)**: Mean, median, std, quantiles, range
- **Formants (F1-F5)**: Frequency, bandwidths, slopes, accelerations
- **Voice Quality**: Jitter (local, RAP, PPQ5, DDP), Shimmer (local, dB, APQ)
- **Harmonicity**: HNR mean and std
- **Intensity**: Mean, std, coefficient of variation
- **MFCCs**: Mean, variance, slopes for 14 coefficients
- **Temporal Complexity**: Trajectory shape complexity measures

### Model
- **Algorithm**: Logistic Regression
- **Features**: 115 acoustic features
- **Output**: Binary classification (Disease / No Disease)
- **Accuracy**: Pre-trained on VOC-ALS dataset (763 samples)

## Limitations

- Maximum file size: 50 MB
- Supported format: `.wav` only
- Processing time: 10-30 seconds depending on audio length and system performance
- Requires stable audio quality for accurate feature extraction

## Troubleshooting

### Model Not Found
Ensure `logistic_regression_model.pkl` is in the same directory as `app.py`.

### Parselmouth Installation Issues
On Windows, you may need to install pre-built wheels:
```bash
pip install parselmouth --only-binary :all:
```

### Port Already in Use
If port 5000 is occupied, modify the port in `app.py`:
```python
app.run(debug=True, port=5001)
```

## Development

To enable debug mode and hot-reload:
```python
app.run(debug=True)
```

To disable debug mode for production:
```python
app.run(debug=False)
```

## Performance Notes

- Feature extraction is the bottleneck (~5-20 seconds per file)
- Model prediction is instant (<100ms)
- Parselmouth utilizes Praat for accurate acoustic analysis
- Multi-threaded processing can be enabled for production deployment

## Future Enhancements

- Support for batch processing
- Multiple model selection (SVM, Random Forest)
- Audio preprocessing options
- Result export to CSV
- Prediction history
- Advanced visualization of acoustic features

## License

This project is part of the ALS Speech Disorder Classification research initiative.

## Contact & Support

For issues or questions, please refer to the main project documentation.
