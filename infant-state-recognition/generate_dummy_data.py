import os
import numpy as np
import scipy.io.wavfile as wavfile

def generate_noise(duration=2.0, sample_rate=22050, freq_mod=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate some basic signal with noise
    signal = np.sin(2 * np.pi * 440 * freq_mod * t) + np.random.normal(0, 0.5, len(t))
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal

def main():
    classes = {
        'crying': 1.5,
        'hungry': 1.0,
        'discomfort': 0.8,
        'sleeping': 0.2
    }
    
    base_dir = "data/raw"
    os.makedirs(base_dir, exist_ok=True)
    
    samples_per_class = 20
    
    for cls, freq_mod in classes.items():
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(samples_per_class):
            signal = generate_noise(duration=2.0, freq_mod=freq_mod)
            # Add some randomness to pitch
            signal += np.random.normal(0, 0.1, len(signal))
            signal = np.int16(signal * 32767)
            wavfile.write(os.path.join(cls_dir, f"sample_{i:03d}.wav"), 22050, signal)
            
    print("Dummy dataset generated successfully in 'data/raw/'.")

if __name__ == "__main__":
    main()
