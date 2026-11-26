from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from config import BLOCK_SIZE
from signal_processor import SignalProcessor


def analyze_file(path, figure=None):
    path = Path(path)
    if path.suffix.lower() not in {".mp3", ".wav"}:
        raise ValueError("MP3 vagy WAV fajlt adj meg.")
    samples, sample_rate = librosa.load(path, sr=None, mono=True)
    processor = SignalProcessor(sample_rate)
    measurement = processor.process(samples)
    peak_rms = 0.0
    for start in range(0, samples.size, BLOCK_SIZE):
        block = samples[start:start + BLOCK_SIZE]
        block_rms = float(np.sqrt(np.mean(block ** 2, dtype=np.float64)))
        peak_rms = max(peak_rms, block_rms)

    peak_db = processor.rms_to_db(peak_rms)

    summary = (f"Atlagos szint: {measurement.db:.1f} dB | "
               f"Csucs: {peak_db:.1f} dB | Dominans frekvencia: {measurement.dominant_hz:.1f} Hz")

    standalone = figure is None
    if standalone:
        figure = plt.figure(figsize=(10, 7), layout="tight")
    figure.clear()
    wave, spectrum = figure.subplots(2, 1)
    wave.plot(np.arange(samples.size) / sample_rate, samples)
    wave.set(title="Hullamforma", xlabel="Ido (s)", ylabel="Amplitudo")
    spectrum.plot(measurement.frequencies, measurement.spectrum)
    spectrum.set(title="FFT spektrum", xlabel="Frekvencia (Hz)", ylabel="FFT amplitudo")
    figure.suptitle(path.name)
    if standalone:
        print(summary)
        try:
            plt.show()
        finally:
            plt.close(figure)
    return summary
