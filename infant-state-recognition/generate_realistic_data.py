"""
Realistic Infant Cry Simulator — 5 Kaggle Classes
Matches: warcoder/infant-cry-audio-corpus
  belly_pain | burping | discomfort | hungry | tired

Replicates the Kaggle imbalance:
  hungry=382  discomfort=27  tired=24  belly_pain=16  burping=8

Acoustic models based on published infant cry research:
  • hungry    – periodic wailing cry,  F0 350-550 Hz
  • discomfort– fussing + whine,       F0 250-450 Hz  (overlaps hungry)
  • belly_pain– high-pitched scream,   F0 500-800 Hz  + irregular bursts
  • tired     – low whimper,           F0 200-400 Hz  + yawn-like envelope
  • burping   – low grunt + rumble,    F0 80-200 Hz   + burst-decay pattern
"""

import os, numpy as np, scipy.io.wavfile as wavfile

SR       = 22050
DURATION = 3.0
BASE_DIR = "data/raw"

# Match Kaggle counts exactly
CLASS_COUNTS = {
    'hungry':     382,
    'discomfort':  27,
    'tired':       24,
    'belly_pain':  16,
    'burping':      8,
}

np.random.seed(None)


# ── acoustic primitives ───────────────────────────────────────

def voiced(t, f0, n_harm=8, jitter=0.02, shimmer=0.05):
    """Multi-harmonic voiced sound with pitch jitter."""
    f0_track = f0 * (1 + jitter * np.sin(2*np.pi*0.4*t)
                     + 0.5*jitter * np.random.randn())
    phase = np.cumsum(2*np.pi * f0_track / SR)
    sig   = np.zeros_like(t)
    for k in range(1, n_harm+1):
        amp = (1/k) * (1 + shimmer * np.random.randn())
        sig += amp * np.sin(k * phase + np.random.uniform(0, 0.1))
    return sig

def am(t, rate, depth=0.6, phase_offset=0.0):
    """Amplitude modulation."""
    return 1 - depth*0.5*(1 - np.cos(2*np.pi*rate*t + phase_offset))

def add_noise(sig, snr_db):
    p = np.mean(sig**2) + 1e-10
    return sig + np.sqrt(p / 10**(snr_db/10)) * np.random.randn(len(sig))

def normalize(sig):
    return sig / (np.max(np.abs(sig)) + 1e-8)


# ── class generators ─────────────────────────────────────────

def gen_hungry():
    """Periodic wailing cry with rhythmic pauses. F0: 350-550 Hz."""
    t   = np.linspace(0, DURATION, int(SR*DURATION))
    f0  = np.random.uniform(350, 550)
    sig = voiced(t, f0, n_harm=np.random.randint(6,11),
                 jitter=np.random.uniform(0.01,0.05),
                 shimmer=np.random.uniform(0.03,0.10))
    sig *= am(t, np.random.uniform(2.0,3.5), depth=np.random.uniform(0.5,0.75))
    # 1-2 brief pauses (gasp before next cry)
    for _ in range(np.random.randint(1,3)):
        s = np.random.randint(0, len(t)-SR//4)
        sig[s:s + np.random.randint(SR//10, SR//5)] *= 0.05
    return normalize(add_noise(sig, np.random.uniform(15,30)))


def gen_discomfort():
    """Fussy whine — overlaps hungry F0 range. F0: 250-450 Hz."""
    t   = np.linspace(0, DURATION, int(SR*DURATION))
    f0  = np.random.uniform(250, 450)         # overlaps hungry on purpose
    sig = voiced(t, f0, n_harm=np.random.randint(4,8),
                 jitter=np.random.uniform(0.02,0.07),
                 shimmer=np.random.uniform(0.05,0.12))
    sig *= am(t, np.random.uniform(1.0,2.5), depth=np.random.uniform(0.4,0.65))
    # irregular amplitude spurts (fussing)
    n_samp = len(t)
    mask = np.ones(n_samp)*0.35
    for _ in range(np.random.randint(3,7)):
        s = np.random.randint(0, n_samp-SR//3)
        mask[s:s+np.random.randint(SR//8,SR//3)] = np.random.uniform(0.7,1.0)
    sig *= mask
    return normalize(add_noise(sig, np.random.uniform(10,22)))


def gen_belly_pain():
    """High-pitched scream + irregular pain bursts. F0: 500-800 Hz."""
    t   = np.linspace(0, DURATION, int(SR*DURATION))
    f0  = np.random.uniform(500, 800)          # highest pitch class
    sig = voiced(t, f0, n_harm=np.random.randint(8,14),
                 jitter=np.random.uniform(0.03,0.08),
                 shimmer=np.random.uniform(0.05,0.15))
    # pain burst: sudden loud scream then fade
    burst_start = np.random.randint(0, len(t)//2)
    burst_len   = np.random.randint(SR//3, SR)
    env = np.ones(len(t)) * 0.4
    env[burst_start:burst_start+burst_len] = np.linspace(
        1.0, 0.3, min(burst_len, len(t)-burst_start))
    sig *= env
    # slight glottal pulse noise
    sig += 0.08 * np.random.randn(len(t))
    return normalize(add_noise(sig, np.random.uniform(12,25)))


def gen_tired():
    """Low-energy whimper + yawn envelope. F0: 200-400 Hz."""
    t   = np.linspace(0, DURATION, int(SR*DURATION))
    f0  = np.random.uniform(200, 400)
    sig = voiced(t, f0, n_harm=np.random.randint(3,7),
                 jitter=np.random.uniform(0.02,0.06),
                 shimmer=np.random.uniform(0.04,0.10))
    # yawn-like slow AM + descending pitch
    yawn_env = np.exp(-1.5 * np.linspace(0, 1, len(t)))   # fade-out
    sig *= yawn_env * am(t, np.random.uniform(0.4,1.0),
                          depth=np.random.uniform(0.3,0.6))
    # soft — low amplitude
    sig *= np.random.uniform(0.3, 0.6)
    return normalize(add_noise(sig, np.random.uniform(8,18)))


def gen_burping():
    """Low grunt + rumble + single burst. F0: 80-200 Hz."""
    t   = np.linspace(0, DURATION, int(SR*DURATION))
    f0  = np.random.uniform(80, 200)           # lowest pitch class
    sig = voiced(t, f0, n_harm=np.random.randint(2,6),
                 jitter=np.random.uniform(0.05,0.12),
                 shimmer=np.random.uniform(0.08,0.18))
    # single abrupt burst (the actual burp event)
    burst_t   = np.random.uniform(0.3, DURATION-0.5)
    burst_s   = int(burst_t * SR)
    burst_len = np.random.randint(SR//12, SR//5)
    burst_env = np.hanning(burst_len)
    end = min(burst_s + burst_len, len(t))
    seg = burst_env[:end-burst_s]
    sig[burst_s:end] = sig[burst_s:end]*0.1 + np.random.randn(end-burst_s)*0.3*seg
    # low rumble
    sig += 0.15 * np.sin(2*np.pi*np.random.uniform(40,80)*t)
    sig[:burst_s]   *= 0.2    # quiet before burst
    sig[end:]        *= 0.15  # quiet after
    return normalize(add_noise(sig, np.random.uniform(6,16)))


GENERATORS = {
    'hungry':     gen_hungry,
    'discomfort': gen_discomfort,
    'belly_pain': gen_belly_pain,
    'tired':      gen_tired,
    'burping':    gen_burping,
}


def main():
    print("Generating realistic 5-class infant cry dataset...")
    print(f"Matching Kaggle imbalance: {CLASS_COUNTS}")
    total = sum(CLASS_COUNTS.values())
    print(f"Total samples: {total}\n")

    for cls, count in CLASS_COUNTS.items():
        cls_dir = os.path.join(BASE_DIR, cls)
        # remove previous files
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.endswith('.wav'):
                    os.remove(os.path.join(cls_dir, f))
        os.makedirs(cls_dir, exist_ok=True)

        gen = GENERATORS[cls]
        for i in range(count):
            sig = gen()
            wavfile.write(os.path.join(cls_dir, f"sample_{i:04d}.wav"),
                          SR, np.int16(sig*32767))

        print(f"  {cls:<14} — {count:>4} samples  ✓")

    print("\nDone!")
    print("Acoustic properties per class:")
    print("  hungry      F0 350-550 Hz  periodic wail + pauses")
    print("  discomfort  F0 250-450 Hz  fussy whine   (overlaps hungry)")
    print("  belly_pain  F0 500-800 Hz  screaming     + pain burst")
    print("  tired       F0 200-400 Hz  low whimper   + yawn fade")
    print("  burping     F0  80-200 Hz  grunt + single burst")
    print("\nClass imbalance (like real Kaggle data):")
    for c,n in sorted(CLASS_COUNTS.items(), key=lambda x:-x[1]):
        bar = '█'*min(n//10+1,40)
        print(f"  {c:<14} {bar} {n}")


if __name__ == "__main__":
    main()
