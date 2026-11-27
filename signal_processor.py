from dataclasses import dataclass

import numpy as np

from config import DB_OFFSET, SAMPLE_RATE


BANDS = {"Mely": (20, 250), "Kozep": (250, 2000), "Magas": (2000, 8000)}


@dataclass
class Measurement:
    rms: float
    db: float
    peak_db: float
    dominant_hz: float
    frequencies: np.ndarray
    spectrum: np.ndarray
    band_power: dict[str, float]

    @property
    def strongest_band(self):
        strongest = max(self.band_power, key=self.band_power.get)
        return strongest if self.band_power[strongest] > 0 else "-"


class SignalProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.peak_db = 0.0

    @staticmethod
    def rms_to_db(rms):
        if rms <= 0:
            return 0.0
        return max(0.0, float(20 * np.log10(rms)) + DB_OFFSET)

    def reset_peak(self):
        self.peak_db = 0.0

    def process(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("Nem ures mono hangblokk szukseges.")

        rms = float(np.sqrt(np.mean(samples ** 2)))
        db = self.rms_to_db(rms)
        self.peak_db = max(self.peak_db, db)

        frequencies = np.fft.rfftfreq(samples.size, 1 / self.sample_rate)
        window = np.hanning(samples.size) if samples.size > 2 else np.ones(samples.size)
        spectrum = np.abs(np.fft.rfft((samples - samples.mean()) * window))
        spectrum[0] = 0
        peak_index = int(np.argmax(spectrum))
        dominant_hz = float(frequencies[peak_index]) if spectrum[peak_index] > 0 else 0.0
        power = spectrum ** 2
        band_power = {}
        for band, (lower, upper) in BANDS.items():
            upper_mask = frequencies <= upper if band == "Magas" else frequencies < upper
            mask = (frequencies >= lower) & upper_mask
            band_power[band] = float(power[mask].sum())

        return Measurement(rms, db, self.peak_db, dominant_hz, frequencies, spectrum, band_power)
