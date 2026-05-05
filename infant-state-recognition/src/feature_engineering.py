import librosa
import numpy as np
import os
from src.preprocessing import preprocess_pipeline

def extract_features(audio, sr=22050):
    """
    Extract the legacy 53-dim ML feature vector used by the current saved
    sklearn artifacts in this repo.

    Layout:
    - MFCC mean (40)
    - Zero crossing rate mean (1)
    - Chroma mean (12)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr.T, axis=0)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    return np.hstack([mfcc_mean, zcr_mean, chroma_mean])


def extract_extended_features(audio, sr=22050):
    """
    Extract the newer 142-dim feature vector described in the project report.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr.T, axis=0)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

    return np.hstack(
        [
            mfcc_mean,
            mfcc_std,
            delta_mfcc,
            zcr_mean,
            chroma_mean,
            contrast,
            rolloff,
            rms,
        ]
    )


def extract_features_for_dim(audio, sr=22050, n_features=None):
    """
    Match feature extraction to the loaded artifact dimensionality.
    """
    if n_features == 142:
        return extract_extended_features(audio, sr)
    return extract_features(audio, sr)

def extract_spectrogram(audio, sr=22050, n_mels=128, max_pad_len=100):
    """
    Extract Mel Spectrogram for DL models (CNN).
    Pads or truncates to max_pad_len.
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or truncate
    if mel_spec_db.shape[1] > max_pad_len:
        mel_spec_db = mel_spec_db[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
        
    return mel_spec_db

def load_data(data_dir="data/raw", augment=False, feature_type='ml'):
    """
    Load data from directory.
    feature_type: 'ml' for 1D features, 'dl' for 2D spectrograms.
    Returns X (features), y (labels), and class_mapping.
    """
    classes = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))])
    class_mapping = {label: idx for idx, label in enumerate(classes)}
    
    X = []
    y = []
    
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for val in os.listdir(class_dir):
            if not val.endswith('.wav'):
                continue
            
            file_path = os.path.join(class_dir, val)
            audios, sr = preprocess_pipeline(file_path, augment=augment)
            
            for audio in audios:
                if feature_type == 'ml':
                    features = extract_features(audio, sr)
                else:
                    features = extract_spectrogram(audio, sr)
                X.append(features)
                y.append(class_mapping[label])
                
    return np.array(X), np.array(y), class_mapping
