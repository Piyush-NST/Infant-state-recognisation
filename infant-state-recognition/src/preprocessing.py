import librosa
import numpy as np
import os

def load_audio(file_path, sr=22050):
    """Load an audio file."""
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def normalize_audio(audio):
    """Normalize audio array."""
    return librosa.util.normalize(audio)

def add_noise(audio, noise_factor=0.005):
    """Add random noise to audio."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def pitch_shift(audio, sr, n_steps=2):
    """Shift the pitch of an audio signal."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def preprocess_pipeline(file_path, sr=22050, augment=False):
    """Complete preprocessing pipeline for an audio file."""
    audio, sr = load_audio(file_path, sr)
    audio = normalize_audio(audio)
    
    samples = [audio]
    if augment:
        samples.append(add_noise(audio))
        samples.append(pitch_shift(audio, sr))
        
    return samples, sr
